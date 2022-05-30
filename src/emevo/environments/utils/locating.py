import enum
from typing import Any, Callable, Iterable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

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
            assert False, "Unreachable"


def init_loc_gaussian(mean: ArrayLike, stddev: ArrayLike) -> InitLocFn:
    mean = np.array(mean)
    stddev = np.array(stddev)
    return lambda generator: generator.normal(loc=mean, scale=stddev)


def init_loc_pre_defined(locations: Iterable[NDArray]) -> InitLocFn:
    location_iter = iter(locations)
    return lambda _generator: next(location_iter)


def init_loc_uniform(low: ArrayLike, high: ArrayLike) -> InitLocFn:
    low = np.array(low)
    high = np.array(high)
    return lambda generator: generator.uniform(low=low, high=high)
