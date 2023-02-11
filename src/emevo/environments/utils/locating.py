from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Iterable, Protocol

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray


class Coordinate(Protocol):
    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        ...

    def contains_circle(self, center: ArrayLike, radius: float) -> bool:
        ...

    def uniform(self, generator: Generator) -> NDArray:
        ...


@dataclasses.dataclass
class CircleCoordinate(Coordinate):
    center: tuple[float, float]
    radius: float

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center
        r = self.radius
        return (cx - r, cx + r), (cy - r, cy + r)

    def contains_circle(self, center: ArrayLike, radius: float) -> bool:
        a2b = np.array(center) - np.array(self.center)  # type: ignore
        distance = np.linalg.norm(a2b, ord=2) - radius
        return bool(distance <= self.radius)

    def uniform(self, generator: Generator) -> NDArray:
        low = [0.0, 0.0]
        high = [1.0, 2.0 * np.pi]
        squared_norm, angle = generator.uniform(low=low, high=high)
        radius = self.radius * np.sqrt(squared_norm)
        cx, cy = self.center
        return np.array([radius * np.cos(angle) + cx, radius * np.sin(angle) + cy])


@dataclasses.dataclass
class SquareCoordinate(Coordinate):
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    offset: float

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.xlim, self.ylim

    def contains_circle(self, center: ArrayLike, radius: float) -> bool:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        x, y = np.array(center)
        offset = self.offset + radius
        x_in = xmin + offset <= x and x <= xmax - offset
        y_in = ymin + offset <= y and y <= ymax - offset
        return x_in and y_in

    def uniform(self, generator: Generator) -> NDArray:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = np.array([xmin + self.offset, ymin + self.offset])
        high = np.array([xmax - self.offset, ymax - self.offset])
        return generator.uniform(low=low, high=high)


InitLocFn = Callable[[Generator], NDArray]


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
    mean = np.array(mean)
    stddev = np.array(stddev)
    return lambda generator: generator.normal(loc=mean, scale=stddev)


def init_loc_gaussian_mixture(
    probs: ArrayLike,
    mean_arr: ArrayLike,
    stddev_arr: ArrayLike,
) -> InitLocFn:
    mean_a = np.array(mean_arr)
    stddev_a = np.array(stddev_arr)

    def sample(generator: Generator) -> NDArray:
        i = generator.choice(len(probs), p=probs)
        return generator.normal(loc=mean_a[i], scale=stddev_a[i])

    return sample


def init_loc_pre_defined(locations: Iterable[NDArray]) -> InitLocFn:
    location_iter = iter(locations)
    return lambda _generator: next(location_iter)


def init_loc_uniform(coordinate: Coordinate) -> InitLocFn:
    return lambda generator: coordinate.uniform(generator)
