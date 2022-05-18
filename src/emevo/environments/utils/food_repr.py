"""
Utility functions to write food reproduction code in foraging environments.
"""

import enum

from typing import Any, Callable, List

import numpy as np

from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from emevo.environments.utils.locating import init_loc_gaussian, init_loc_uniform
from emevo.types import Location

ReprNumFn = Callable[[int], int]
ReprLocFn = Callable[[Generator, List[Location]], NDArray]


class ReprNum(str, enum.Enum):
    """Methods to determine the number of foods reproduced."""

    CONSTANT = "constant"
    LOGISTIC = "logistic"

    def __call__(self, *args: Any, **kwargs: Any) -> ReprNumFn:
        if self is ReprNum.CONSTANT:
            return repr_num_constant(*args, **kwargs)
        elif self is ReprNum.LOGISTIC:
            return repr_num_logistic(*args, **kwargs)
        else:
            assert False, "Unreachable"


def repr_num_constant(n_foods: int) -> ReprNumFn:
    return lambda current: max(0, n_foods - current)


def repr_num_logistic(growth_rate: float, capacity: float) -> ReprNumFn:
    def repr_num(n_foods: int) -> int:
        dn_dt = growth_rate * n_foods * (1 - n_foods / capacity)
        return np.rint(dn_dt)

    return repr_num


class ReprLoc(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> ReprLocFn:
        if self is ReprLoc.GAUSSIAN:
            return repr_loc_gaussian(*args, **kwargs)
        elif self is ReprLoc.UNIFORM:
            return repr_loc_uniform(*args, **kwargs)
        else:
            assert False, "Unreachable"


def repr_loc_gaussian(mean: ArrayLike, stddev: ArrayLike) -> ReprLocFn:
    fn = init_loc_gaussian(mean, stddev)
    return lambda generator, _locations: fn(generator)


def repr_loc_uniform(low: ArrayLike, high: ArrayLike) -> ReprLocFn:
    fn = init_loc_uniform(low, high)
    return lambda generator, _locations: fn(generator)
