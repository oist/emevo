"""
Utility functions to write food reproduction code in foraging environments.
"""

import enum

from typing import Any, Callable

import numpy as np

FoodReprFn = Callable[[int], int]


class ReprMethods(str, enum.Enum):
    constant = "constant"
    logistic = "logistic"

    def __call__(self, *args: Any, **kwargs: Any) -> FoodReprFn:
        if self.value == "constant":
            return constant_repr(*args, **kwargs)
        elif self.value == "logistic":
            return logistic_repr(*args, **kwargs)
        else:
            assert False, "Unreachable"


def constant_repr(n_foods: int) -> FoodReprFn:
    return lambda current: max(0, n_foods - current)


def logistic_repr(growth_rate: float, capacity: float) -> FoodReprFn:
    def reproduce_fn(n_foods: int) -> int:
        dn_dt = growth_rate * n_foods * (1 - n_foods / capacity)
        return np.rint(dn_dt)

    return reproduce_fn
