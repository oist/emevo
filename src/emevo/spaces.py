"""Similar to gym.spaces.Space, but doesn't have RNG"""
from __future__ import annotations

import abc
from typing import Any, Generic, Iterable, NamedTuple, Sequence, Type, TypeVar

import numpy as np
from numpy.random import Generator
from numpy.typing import DTypeLike, NDArray

INSTANCE = TypeVar("INSTANCE")


class Space(abc.ABC, Generic[INSTANCE]):
    dtype: np.dtype
    shape: tuple[int, ...]

    @abc.abstractmethod
    def contains(self, x: INSTANCE) -> bool:
        pass

    @abc.abstractmethod
    def sample(self, generator: Generator) -> INSTANCE:
        pass

    @abc.abstractmethod
    def flatten(self) -> BoxSpace:
        raise NotImplementedError()


def _short_repr(arr: NDArray) -> str:
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


class BoxSpace(Space[NDArray]):
    """gym.spaces.Box, but without RNG"""

    def __init__(
        self,
        low: int | float | NDArray,
        high: int | float | NDArray,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike = np.float32,
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
            isinstance(other, self.__class__)
            and (self.shape == other.shape)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )

    def flatten(self) -> BoxSpace:
        return self


def get_inf(dtype, sign: str) -> int | float:
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
    value: int | float | NDArray,
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


class DiscreteSpace(Space[int]):
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

    def contains(self, x: int) -> bool:
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
            isinstance(other, self.__class__)
            and self.n == other.n
            and self.start == other.start
        )

    def flatten(self) -> BoxSpace:
        return BoxSpace(low=np.zeros(self.n), high=np.ones(self.n))


class NamedTupleSpace(Space[NamedTuple], Iterable):
    """Space that returns namedtuple of other spaces"""

    def __init__(self, cls: Type[tuple], **spaces_kwargs: Space) -> None:
        assert all(
            [isinstance(s, Space) for s in spaces_kwargs.values()]
        ), "All arguments of NamedTuple space should be a subclass of Space"
        name = cls.__name__
        self._cls = cls
        fields = cls._fields  # type: ignore
        possibly_missing_keys = set(fields)
        for key in spaces_kwargs:
            if key not in fields:
                raise ValueError(f"Invalid key for {name}: {key}")
            possibly_missing_keys.remove(key)
        if len(possibly_missing_keys):
            raise ValueError(f"Missing keys: {list(possibly_missing_keys)}")
        spaces = [(field, spaces_kwargs[field].__class__) for field in fields]
        self._space_cls = NamedTuple(name + "Space", spaces)
        self.spaces = self._space_cls(**spaces_kwargs)
        dtype = self.spaces[0].dtype
        for space in self.spaces:
            if space.dtype != dtype:
                raise ValueError("All dtype of NamedTuple space must be the same")
        self.dtype = dtype
        self.shape = tuple(space.shape for space in self.spaces)

    def sample(self, generator: Generator) -> Any:
        samples = tuple(space.sample(generator) for space in self.spaces)
        return self._cls(*samples)

    def contains(self, x: tuple) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        for instance, space in zip(x, self.spaces):
            if not space.contains(instance):
                return False

        return True

    def __getitem__(self, key: str) -> Space:
        """Get the space that is associated to `key`."""
        return getattr(self.spaces, key)

    def __iter__(self) -> Iterable[Space]:
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `NamedTuple` space."""
        return len(self.spaces)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        spaces = ",".join(
            [f"{k}:{s}" for k, s in zip(self._space_cls._fields, self.spaces)]
        )
        name = self._space_cls.__name__
        return f"{name}({spaces})"

    def flatten(self) -> BoxSpace:
        spaces = [space.flatten() for space in self.spaces]
        low = np.concatenate([space.low for space in spaces])
        high = np.concatenate([space.high for space in spaces])
        return BoxSpace(low=low, high=high)
