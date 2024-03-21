"""Similar to gym.spaces.Space, but for jax"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Generic, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp

INSTANCE = TypeVar("INSTANCE")
DTYPE = TypeVar("DTYPE")


class Space(abc.ABC, Generic[INSTANCE, DTYPE]):
    dtype: DTYPE
    shape: tuple[int, ...]

    @abc.abstractmethod
    def clip(self, x: INSTANCE) -> INSTANCE:
        raise NotImplementedError()

    @abc.abstractmethod
    def contains(self, x: INSTANCE) -> jax.Array:
        pass

    @abc.abstractmethod
    def flatten(self) -> BoxSpace:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, key: chex.PRNGKey) -> INSTANCE:
        pass


def _short_repr(arr: jax.Array) -> str:
    if arr.size != 0 and jnp.min(arr) == jnp.max(arr):
        return str(jnp.min(arr))
    return str(arr)


class BoxSpace(Space[jax.Array, jnp.dtype]):
    """gym.spaces.Box, but without RNG"""

    def __init__(
        self,
        low: int | float | jax.Array,
        high: int | float | jax.Array,
        shape: Sequence[int] | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.dtype = jnp.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
        elif not jnp.isscalar(low):
            shape = low.shape  # type: ignore
        elif not jnp.isscalar(high):
            shape = high.shape  # type: ignore
        else:
            raise ValueError(
                "shape must be provided or inferred from the shapes of low or high"
            )
        assert isinstance(shape, tuple)
        self.shape = shape

        # Capture the boundedness information before replacing jnp.inf with get_inf
        _low = jnp.full(shape, low, dtype=jnp.float32) if jnp.isscalar(low) else low
        self.bounded_below = -jnp.inf < _low  # type: ignore
        _high = jnp.full(shape, high, dtype=jnp.float32) if jnp.isscalar(high) else high
        self.bounded_above = jnp.inf > _high  # type: ignore

        low = _broadcast(low, dtype, shape, inf_sign="-")  # type: ignore
        high = _broadcast(high, dtype, shape, inf_sign="+")  # type: ignore

        assert isinstance(low, jax.Array)
        assert low.shape == shape, "low.shape doesn't match provided shape"
        assert isinstance(high, jax.Array)
        assert high.shape == shape, "high.shape doesn't match provided shape"

        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)
        self._range = self.high - self.low

        self.low_repr = _short_repr(self.low)
        self.high_repr = _short_repr(self.high)

    def is_bounded(self, manner: str = "both") -> bool:
        below = jnp.all(self.bounded_below).item()
        above = jnp.all(self.bounded_above).item()
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def clip(self, x: jax.Array) -> jax.Array:
        return jnp.clip(x, a_min=self.low, a_max=self.high)

    def contains(self, x: jax.Array) -> jax.Array:
        type_ok = jnp.can_cast(x.dtype, self.dtype) and x.shape == self.shape
        value_ok = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return jnp.logical_and(type_ok, value_ok)

    def flatten(self) -> BoxSpace:
        return BoxSpace(low=self.low.flatten(), high=self.high.flatten())

    def sample(self, key: chex.PRNGKey) -> jax.Array:
        low = self.low.astype(jnp.float32)
        if self.dtype.kind == "f":
            high = self.high
        else:
            high = self.high.astype(jnp.float32) + 1.0
        key1, key2, key3, key4 = jax.random.split(key, 4)
        sample = jnp.where(
            # Bounded
            jnp.logical_and(self.bounded_below, self.bounded_above),
            jax.random.uniform(key1, minval=low, maxval=high, shape=self.shape),
            jnp.where(
                self.bounded_below,
                # Low bounded
                low + jax.random.exponential(key2, shape=self.shape),
                jnp.where(
                    self.bounded_above,
                    # High bounded
                    high - jax.random.exponential(key3, shape=self.shape),
                    # Unbounded
                    jax.random.normal(key4, shape=self.shape),
                ),
            ),
        )

        if self.dtype.kind == "i":
            return jnp.floor(sample).astype(self.dtype)
        else:
            return sample.astype(self.dtype)

    def normalize(self, unnormalized: jax.Array) -> jax.Array:
        return (unnormalized - self.low) / self._range

    def sigmoid_scale(self, array: jax.Array) -> jax.Array:
        return self._range * jax.nn.sigmoid(array) + self.low

    def __repr__(self) -> str:
        return f"Box({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"


def get_inf(dtype, sign: str) -> int | float:
    """Returns an infinite that doesn't break things.
    Args:
        dtype: An `jnp.dtype`
        sign (str): must be either `"+"` or `"-"`
    """
    if jnp.dtype(dtype).kind == "f":
        if sign == "+":
            return jnp.inf
        elif sign == "-":
            return -jnp.inf
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    elif jnp.dtype(dtype).kind == "i":
        if sign == "+":
            return jnp.iinfo(dtype).max - 2
        elif sign == "-":
            return jnp.iinfo(dtype).min + 2
        else:
            raise TypeError(f"Unknown sign {sign}, use either '+' or '-'")
    else:
        raise ValueError(f"Unknown dtype {dtype} for infinite bounds")


def _broadcast(
    value: int | float | jax.Array,
    dtype,
    shape: tuple[int, ...],
    inf_sign: str,
) -> jax.Array:
    """Handle infinite bounds and broadcast at the same time if needed."""
    if jnp.isscalar(value):
        value = get_inf(dtype, inf_sign) if jnp.isinf(value) else value  # type: ignore
        value = jnp.full(shape, value, dtype=dtype)
    else:
        assert isinstance(value, jax.Array)
        isinf = jnp.isinf(value)
        if jnp.any(isinf):
            # create new array with dtype, but maintain old one to preserve jnp.inf
            value = jnp.where(isinf, get_inf(dtype, inf_sign), value.astype(dtype))
    return value


class DiscreteSpace(Space[jax.Array, jnp.dtype]):
    """gym.spaces.Discrete, but without RNG"""

    def __init__(self, n: int, start: int = 0) -> None:
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, jnp.integer))
        self.dtype = jnp.dtype(int)
        self.shape = ()
        self.n = n
        self.start = start

    def clip(self, x: jax.Array) -> jax.Array:
        return jnp.clip(x, a_min=self.start, a_max=self.start + self.n)

    def contains(self, x: jax.Array) -> jax.Array:
        """Return boolean specifying if x is a valid member of this space."""
        return jnp.logical_and(self.start <= x, x < self.start + self.n)

    def flatten(self) -> BoxSpace:
        return BoxSpace(low=jnp.zeros(self.n), high=jnp.ones(self.n))

    def sample(self, key: chex.PRNGKey) -> jax.Array:
        rn = jax.random.randint(key, shape=self.shape, minval=0, maxval=self.n)
        return rn.item() + self.start

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


class NamedTupleSpace(Space[NamedTuple, tuple[jnp.dtype, ...]], Iterable):
    """Space that returns namedtuple of other spaces"""

    def __init__(self, cls: type[tuple], **spaces_kwargs: Space) -> None:
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
        self._space_cls = NamedTuple(
            name + "Space",
            tuple((field, spaces_kwargs[field].__class__) for field in fields),
        )
        self.spaces = self._space_cls(**spaces_kwargs)
        self.dtype = tuple(s.dtype for s in self.spaces)
        self.shape = tuple(space.shape for space in self.spaces)

    def clip(self, x: tuple) -> Any:
        clipped = [space.clip(value) for value, space in zip(x, self.spaces)]
        return self._cls(*clipped)

    def contains(self, x: NamedTuple) -> jax.Array:
        """Return boolean specifying if x is a valid member of this space."""
        contains = [space.contains(instance) for instance, space in zip(x, self.spaces)]
        return jnp.all(jnp.array(contains))

    def flatten(self) -> BoxSpace:
        spaces = [space.flatten() for space in self.spaces]
        low = jnp.concatenate([space.low for space in spaces])
        high = jnp.concatenate([space.high for space in spaces])
        return BoxSpace(low=low, high=high)

    def sample(self, key: chex.PRNGKey) -> Any:
        keys = jax.random.split(key, len(self.spaces))
        samples = [space.sample(key) for space, key in zip(self.spaces, keys)]
        return self._cls(*samples)

    def __getitem__(self, key: str) -> Space:
        """Get the space that is associated to `key`."""
        return getattr(self.spaces, key)

    def __iter__(self) -> Iterator[Space]:
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
