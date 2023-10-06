from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any, Callable, Protocol

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


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

    def contains_circle(self, center: jax.Array, radius: jax.Array) -> bool:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = jnp.array([xmin, ymin]) + radius
        high = jnp.array([xmax, ymax]) - radius
        return jnp.logical_and(low <= center, center <= high)

    def uniform(self, key: chex.PRNGKey) -> jax.Array:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = jnp.array([xmin, ymin])
        high = jnp.array([xmax, ymax])
        return jax.random.uniform(key, shape=(2,), minval=low, maxval=high)


InitLocFn = Callable[[chex.PRNGKey], jax.Array]


class InitLoc(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    PRE_DIFINED = "pre-defined"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> InitLocFn:
        if self is InitLoc.GAUSSIAN:
            return init_loc_gaussian(*args, **kwargs)
        elif self is InitLoc.PRE_DIFINED:
            return init_loc_pre_defined(*args, **kwargs)
        elif self is InitLoc.UNIFORM:
            return init_loc_uniform(*args, **kwargs)
        else:
            raise AssertionError("Unreachable")


def init_loc_gaussian(mean: ArrayLike, stddev: ArrayLike) -> InitLocFn:
    mean_a = jnp.array(mean)
    std_a = jnp.array(stddev)
    shape = mean_a.shape
    return lambda key: jax.random.normal(key, shape=shape) * std_a + mean_a


def init_loc_gaussian_mixture(
    probs: ArrayLike,
    mean_arr: ArrayLike,
    stddev_arr: ArrayLike,
) -> InitLocFn:
    mean_a = jnp.array(mean_arr)
    stddev_a = jnp.array(stddev_arr)
    probs_a = jnp.array(probs)
    n = probs_a.shape[0]

    def sample(key: chex.PRNGKey) -> jax.Array:
        k1, k2 = jax.random.split(key)
        i = jax.random.choice(k1, n, p=probs)
        mi, si = mean_a[i], stddev_a[i]
        return jax.random.normal(k2, shape=mean_a.shape[1:]) * si + mi

    return sample


def init_loc_pre_defined(locations: Iterable[jax.Array]) -> InitLocFn:
    location_iter = iter(locations)
    return lambda _key: next(location_iter)


def init_loc_uniform(coordinate: Coordinate) -> InitLocFn:
    return lambda key: coordinate.uniform(key)
