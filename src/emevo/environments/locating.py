from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any, Callable, Protocol

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

Self = Any


class Coordinate(Protocol):
    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        ...

    def contains_circle(self, center: jax.Array, radius: jax.Array) -> jax.Array:
        ...

    def uniform(self, key: chex.PRNGKey) -> jax.Array:
        ...


@dataclasses.dataclass
class CircleCoordinate(Coordinate):
    center: tuple[float, float]
    radius: float

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center
        r = self.radius
        return (cx - r, cx + r), (cy - r, cy + r)

    def contains_circle(self, center: jax.Array, radius: jax.Array) -> jax.Array:
        a2b = center - jnp.array(self.center)
        distance = jnp.linalg.norm(a2b, ord=2)
        return distance + radius <= self.radius

    def uniform(self, key: chex.PRNGKey) -> jax.Array:
        low = jnp.array([0.0, 0.0])
        high = jnp.array([1.0, 2.0 * jnp.pi])
        squared_norm, angle = jax.random.uniform(
            key,
            shape=(2,),
            minval=low,
            maxval=high,
        )
        radius = self.radius * jnp.sqrt(squared_norm)
        cx, cy = self.center
        return jnp.array([radius * jnp.cos(angle) + cx, radius * jnp.sin(angle) + cy])


@dataclasses.dataclass
class SquareCoordinate(Coordinate):
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.xlim, self.ylim

    def contains_circle(self, center: jax.Array, radius: jax.Array) -> jax.Array:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = jnp.array([xmin, ymin]) + radius
        high = jnp.array([xmax, ymax]) - radius
        return jnp.logical_and(jnp.all(low <= center), jnp.all(center <= high))

    def uniform(self, key: chex.PRNGKey) -> jax.Array:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = jnp.array([xmin, ymin])
        high = jnp.array([xmax, ymax])
        return jax.random.uniform(key, shape=(2,), minval=low, maxval=high)


@chex.dataclass
class LocatingState:
    n_produced: jax.Array

    def increment(self) -> Self:
        return self.replace(n_produced=self.n_produced + 1)


LocatingFn = Callable[[chex.PRNGKey, LocatingState], jax.Array]


class Locating(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian-mixture"
    PERIODIC = "periodic"
    SWITCHING = "switching"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[LocatingFn, LocatingState]:
        state = LocatingState(n_produced=jnp.array(0, dtype=jnp.int32))
        if self is Locating.GAUSSIAN:
            return loc_gaussian(*args, **kwargs), state
        elif self is Locating.GAUSSIAN_MIXTURE:
            return loc_gaussian_mixture(*args, **kwargs), state
        elif self is Locating.PERIODIC:
            return LocPeriodic(*args, **kwargs), state
        elif self is Locating.UNIFORM:
            return loc_uniform(*args, **kwargs), state
        elif self is Locating.SWITCHING:
            return LocSwitching(*args, **kwargs), state
        else:
            raise AssertionError("Unreachable")


def loc_gaussian(mean: ArrayLike, stddev: ArrayLike) -> LocatingFn:
    mean_a = jnp.array(mean)
    std_a = jnp.array(stddev)
    shape = mean_a.shape
    return lambda key, _state: jax.random.normal(key, shape=shape) * std_a + mean_a


def loc_gaussian_mixture(
    probs: ArrayLike,
    mean_arr: ArrayLike,
    stddev_arr: ArrayLike,
) -> LocatingFn:
    mean_a = jnp.array(mean_arr)
    stddev_a = jnp.array(stddev_arr)
    probs_a = jnp.array(probs)
    n = probs_a.shape[0]

    def sample(key: chex.PRNGKey, _state: LocatingState) -> jax.Array:
        k1, k2 = jax.random.split(key)
        i = jax.random.choice(k1, n, p=probs_a)
        mi, si = mean_a[i], stddev_a[i]
        return jax.random.normal(k2, shape=mean_a.shape[1:]) * si + mi

    return sample


def loc_uniform(coordinate: Coordinate) -> LocatingFn:
    return lambda key, _state: coordinate.uniform(key)


class LocPeriodic:
    def __init__(self, *locations: Iterable[ArrayLike]) -> None:
        self._locations = jnp.array(list(locations))
        self._n = self._locations.shape[0]

    def __call__(self, key: chex.PRNGKey, state: LocatingState) -> jax.Array:
        count = state.n_produced + 1
        return self._locations[count % self._n]


class LocSwitching:
    def __init__(
        self,
        interval: int,
        *loc_fns: Iterable[tuple[str, ...] | LocatingFn],
    ) -> None:
        locfn_list = []
        for fn_or_base in loc_fns:
            if callable(fn_or_base):
                locfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                locfn_list.append(Locating(name)(*args))
        self._locfn_list = locfn_list
        self._interval = interval
        self._n = len(locfn_list)

    def __call__(self, key: chex.PRNGKey, state: LocatingState) -> jax.Array:
        count = state.n_produced + 1
        index = (count // self._interval) % self._n
        return jax.lax.switch(index, self._locfn_list, key)
