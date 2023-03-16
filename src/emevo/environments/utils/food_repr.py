"""
Utility functions to write food reproduction code in foraging environments.
"""


import dataclasses
import enum
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from emevo.environments.utils.locating import (
    InitLocFn,
    init_loc_gaussian,
    init_loc_gaussian_mixture,
    init_loc_pre_defined,
    init_loc_uniform,
)

_Location = ArrayLike


class ReprNumFn(Protocol):
    initial: int

    def __call__(self, current_num: int) -> int:
        ...


@dataclasses.dataclass(frozen=True)
class ReprNumConstant:
    initial: int

    def __call__(self, current_num: int) -> int:
        return max(0, self.initial - current_num)


@dataclasses.dataclass(frozen=True)
class ReprNumLogistic:
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, current_num: int) -> int:
        dn_dt = self.growth_rate * current_num * (1 - current_num / self.capacity)
        return int(np.rint(dn_dt))


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
            raise AssertionError("Unreachable")


ReprLocFn = Callable[[Generator, Sequence[_Location]], NDArray]


def _wrap_initloc(fn: InitLocFn) -> ReprLocFn:
    return lambda generator, _locations: fn(generator)


class ReprLocSwitching:
    def __init__(
        self,
        interval: int,
        *reprloc_fns: tuple[tuple[str, ...] | ReprLocFn],
    ) -> None:
        locfn_list = []
        for fn_or_base in reprloc_fns:
            if callable(fn_or_base):
                locfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                print(fn_or_base, name)
                locfn_list.append(ReprLoc(name)(*args))
        self._locfn_list = locfn_list
        self._interval = interval
        self._count = 0
        self._current = 0

    def __call__(self, generator: Generator, loc: Sequence[_Location]) -> NDArray:
        self._count += 1
        if self._count % self._interval == 0:
            self._current = (self._current + 1) % len(self._locfn_list)
        return self._locfn_list[self._current](generator, loc)


class ReprLoc(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian-mixture"
    PRE_DIFINED = "pre-defined"
    SWITCHING = "switching"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> ReprLocFn:
        if self is ReprLoc.GAUSSIAN:
            return _wrap_initloc(init_loc_gaussian(*args, **kwargs))
        elif self is ReprLoc.GAUSSIAN_MIXTURE:
            return _wrap_initloc(init_loc_gaussian_mixture(*args, **kwargs))
        elif self is ReprLoc.PRE_DIFINED:
            return _wrap_initloc(init_loc_pre_defined(*args, **kwargs))
        elif self is ReprLoc.SWITCHING:
            return ReprLocSwitching(*args, **kwargs)
        elif self is ReprLoc.UNIFORM:
            return _wrap_initloc(init_loc_uniform(*args, **kwargs))
        else:
            raise AssertionError("Unreachable")
