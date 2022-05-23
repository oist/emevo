"""
Utility functions to write food reproduction code in foraging environments.
"""

import abc
import dataclasses
import enum

from typing import Any, Callable, Sequence

import numpy as np

from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from emevo.environments.utils.locating import init_loc_gaussian, init_loc_uniform

_Location = ArrayLike


class ReprNumFn(abc.ABC):
    initial: int

    @abc.abstractmethod
    def __call__(self, current_num: int) -> int:
        pass


@dataclasses.dataclass
class ReprNumConstant(ReprNumFn):
    initial: int

    def __call__(self, current_num: int) -> int:
        return max(0, self.initial - current_num)


@dataclasses.dataclass
class ReprNumLogistic(ReprNumFn):
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, current_num: int) -> int:
        dn_dt = self.growth_rate * current_num * (1 - current_num / self.capacity)
        return np.rint(dn_dt)


class ReprNum(str, enum.Enum):
    """Methods to determine the number of foods reproduced."""

    CONSTANT = "constant"
    LOGISTIC = "logistic"

    def __call__(self, *args: Any, **kwargs: Any) -> ReprNumFn:
        if self is ReprNum.CONSTANT:
            return ReprNumConstant(*args, **kwargs)
        elif self is ReprNum.LOGISTIC:
            return ReprNumLogistic(*args, **kwargs)
        else:
            assert False, "Unreachable"


ReprLocFn = Callable[[Generator, Sequence[_Location]], NDArray]


class ReprLoc(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> ReprLocFn:
        if self is ReprLoc.GAUSSIAN:
            fn = init_loc_gaussian(*args, **kwargs)
            return lambda generator, _locations: fn(generator)
        elif self is ReprLoc.UNIFORM:
            fn = init_loc_uniform(*args, **kwargs)
            return lambda generator, _locations: fn(generator)
        else:
            assert False, "Unreachable"
