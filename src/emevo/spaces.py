"""Similar to gym.spaces.Space, but doesn't have RNG"""
import abc

from typing import (
    Dict as DictType,
    Generic,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from numpy.random import Generator
from numpy.typing import NDArray

INSTANCE = TypeVar("INSTANCE")


class Space(abc.ABC, Generic[INSTANCE]):
    dtype: np.dtype
    shape: Tuple[int, ...]

    @abc.abstractmethod
    def contains(self, x: INSTANCE) -> bool:
        pass

    @abc.abstractmethod
    def sample(self, generator: Generator) -> INSTANCE:
        pass


def _short_repr(arr: NDArray) -> str:
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


class Box(Space[NDArray]):
    """gym.spaces.Box, but without RNG"""

    def __init__(
        self,
        low: Union[SupportsFloat, NDArray],
        high: Union[SupportsFloat, NDArray],
        shape: Optional[Sequence[int]] = None,
        dtype: Type = np.float32,
    ) -> None:
        self.dtype = np.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
        elif not np.isscalar(low):
            shape = low.shape  # type: ignore
        elif not np.isscalar(high):
            shape = high.shape  # type: ignore
        else:
            raise ValueError(
                "shape must be provided or inferred from the shapes of low or high"
            )
        assert isinstance(shape, tuple)
        self.shape = shape

        # Capture the boundedness information before replacing np.inf with get_inf
        _low = np.full(shape, low, dtype=float) if np.isscalar(low) else low
        self.bounded_below = -np.inf < _low  # type: ignore
        _high = np.full(shape, high, dtype=float) if np.isscalar(high) else high
        self.bounded_above = np.inf > _high  # type: ignore

        low = _broadcast(low, dtype, shape, inf_sign="-")  # type: ignore
        high = _broadcast(high, dtype, shape, inf_sign="+")  # type: ignore

        assert isinstance(low, np.ndarray)
        assert low.shape == shape, "low.shape doesn't match provided shape"
        assert isinstance(high, np.ndarray)
        assert high.shape == shape, "high.shape doesn't match provided shape"

        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)

        self.low_repr = _short_repr(self.low)
        self.high_repr = _short_repr(self.high)

    def is_bounded(self, manner: str = "both") -> bool:
        below = bool(np.all(self.bounded_below))
        above = bool(np.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self, generator: Generator) -> NDArray:
        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = generator.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            generator.exponential(size=low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )
        sample[upp_bounded] = (
            -generator.exponential(size=upp_bounded[upp_bounded].shape)
            + self.high[upp_bounded]
        )
        sample[bounded] = generator.uniform(
            low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)
        return sample.astype(self.dtype)

    def contains(self, x: NDArray) -> bool:
        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def __repr__(self) -> str:
        return f"Box({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )


def get_inf(dtype, sign: str) -> SupportsFloat:
    """Returns an infinite that doesn't break things.
    Args:
        dtype: An `np.dtype`
        sign (str): must be either `"+"` or `"-"`
    """
    if np.dtype(dtype).kind == "f":
        if sign == "+":
            return np.inf
        elif sign == "-":
            return -np.inf
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    elif np.dtype(dtype).kind == "i":
        if sign == "+":
            return np.iinfo(dtype).max - 2
        elif sign == "-":
            return np.iinfo(dtype).min + 2
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    else:
        raise ValueError(f"Unknown dtype {dtype} for infinite bounds")


def _broadcast(
    value: Union[SupportsFloat, NDArray],
    dtype,
    shape: tuple[int, ...],
    inf_sign: str,
) -> NDArray:
    """Handle infinite bounds and broadcast at the same time if needed."""
    if np.isscalar(value):
        value = get_inf(dtype, inf_sign) if np.isinf(value) else value  # type: ignore
        value = np.full(shape, value, dtype=dtype)
    else:
        assert isinstance(value, np.ndarray)
        if np.any(np.isinf(value)):
            # create new array with dtype, but maintain old one to preserve np.inf
            temp = value.astype(dtype)
            temp[np.isinf(value)] = get_inf(dtype, inf_sign)
            value = temp
    return value


class Discrete(Space[int]):
    """gym.spaces.Discrete, but without RNG"""

    def __init__(self, n: int, start: int = 0) -> None:
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, np.integer))
        self.dtype = np.dtype(int)
        self.shape = ()
        self.n = int(n)
        self.start = int(start)

    def sample(self, generator: Generator) -> int:
        return int(self.start + generator.integers(self.n))

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)  # type: ignore
        else:
            return False
        return self.start <= as_int < self.start + self.n

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return "Discrete(%d, start=%d)" % (self.n, self.start)
        return "Discrete(%d)" % self.n

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )


class Dict(Space[DictType[str, Space]], Mapping):
    """gym.spaces.Dict, but without RNG"""

    def __init__(
        self,
        spaces: Optional[dict[str, Space]] = None,
        **spaces_kwargs: Space,
    ) -> None:
        assert (spaces is None) or (
            not spaces_kwargs
        ), "Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)"

        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            try:
                spaces = OrderedDict(sorted(spaces.items()))
            except TypeError:  # raise when sort by different types of keys
                spaces = OrderedDict(spaces.items())
        if isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)

        assert isinstance(spaces, OrderedDict), "spaces must be a dictionary"

        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(
                space, Space
            ), "Values of the dict should be instances of gym.Space"
        super().__init__(
            None, None, seed  # type: ignore
        )  # None for shape and dtype, since it'll require special handling

    def sample(self, generator: Generator) -> dict:
        return OrderedDict(
            [(k, space.sample(generator)) for k, space in self.spaces.items()]
        )

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def __setitem__(self, key, value):
        """Set the space that is associated to `key`."""
        self.spaces[key] = value

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            "Dict("
            + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )
