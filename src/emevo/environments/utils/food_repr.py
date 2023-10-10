"""
Utility functions to write food reproduction code in foraging environments.
"""
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any, Callable, Protocol

import chex
import jax
import jax.numpy as jnp

from emevo.environments.utils.locating import (
    InitLoc,
    InitLocFn,
    init_loc_gaussian,
    init_loc_gaussian_mixture,
    init_loc_pre_defined,
    init_loc_uniform,
)

Self = Any


@chex.dataclass
class FoodNumState:
    current: int
    internal: float

    def appears(self) -> jax.Array:
        return (self.internal - self.current) >= 1.0

    def eaten(self, n: jax.Array) -> Self:
        return self.replace(current=self.current - n, internal=self.internal - n)

    def fail(self, n: jax.Array) -> Self:
        return self.replace(internal=self.internal - n)

    def recover(self, n: jax.Array) -> Self:
        return self.replace(current=self.current + n)


class ReprNumFn(Protocol):
    initial: int

    def __call__(self, state: FoodNumState) -> FoodNumState:
        ...


@dataclasses.dataclass(frozen=True)
class ReprNumConstant:
    initial: int

    def __call__(self, state: FoodNumState) -> FoodNumState:
        diff = jnp.clip(self.initial - state.current, a_min=0)
        state = state.replace(internal=state.internal + diff)
        return state


@dataclasses.dataclass(frozen=True)
class ReprNumLinear:
    initial: int
    dn_dt: float

    def __call__(self, state: FoodNumState) -> FoodNumState:
        # Increase the number of foods by dn_dt
        internal = jnp.clip(state.internal + self.dn_dt, a_max=float(self.initial))
        return state.replace(internal=internal)


@dataclasses.dataclass(frozen=True)
class ReprNumLogistic:
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, state: FoodNumState) -> FoodNumState:
        dn_dt = self.growth_rate * state.internal * (1 - state.internal / self.capacity)
        return state.replace(internal=state.internal + dn_dt)


class ReprNum(str, enum.Enum):
    """Methods to determine the number of foods reproduced."""

    CONSTANT = "constant"
    LINEAR = "linear"
    LOGISTIC = "logistic"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[ReprNumFn,]:
        if len(args) > 0:
            initial = args[0]
        elif "initial" in kwargs:
            initial = kwargs["initial"]
        else:
            raise ValueError("'initial' is required for all ReprNum functions")
        state = FoodNumState(int(initial), float(initial))
        if self is ReprNum.CONSTANT:
            fn = ReprNumConstant(**kwargs)
        elif self is ReprNum.LINEAR:
            fn = ReprNumLinear(**kwargs)
        elif self is ReprNum.LOGISTIC:
            fn = ReprNumLogistic(**kwargs)
        else:
            raise AssertionError("Unreachable")
        return fn, state


@chex.dataclass
class SwitchingState:
    count: int


ReprLocFn = Callable[[chex.PRNGKey, Any], tuple[jax.Array, Any]]


def _wrap_initloc(fn: InitLocFn) -> ReprLocFn:
    return lambda key, _state: (fn(key), _state)


class ReprLocSwitching:
    def __init__(
        self,
        interval: int,
        *initloc_fns: Iterable[tuple[str, ...] | InitLocFn],
    ) -> None:
        locfn_list = []
        for fn_or_base in initloc_fns:
            if callable(fn_or_base):
                locfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                locfn_list.append(InitLoc(name)(*args))
        self._locfn_list = locfn_list
        self._interval = interval
        self._n = len(locfn_list)

    def __call__(self, key: chex.PRNGKey, state: SwitchingState) -> jax.Array:
        count = state.count + 1
        index = (count // self._interval) % self._n
        result = jax.lax.switch(index, self._locfn_list, key)
        return result, state.replace(count=count)


class ReprLoc(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian-mixture"
    PRE_DIFINED = "pre-defined"
    SWITCHING = "switching"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[ReprLocFn, Any]:
        if self is ReprLoc.GAUSSIAN:
            return _wrap_initloc(init_loc_gaussian(*args, **kwargs)), None
        elif self is ReprLoc.GAUSSIAN_MIXTURE:
            return _wrap_initloc(init_loc_gaussian_mixture(*args, **kwargs)), None
        elif self is ReprLoc.PRE_DIFINED:
            return _wrap_initloc(init_loc_pre_defined(*args, **kwargs)), None
        elif self is ReprLoc.SWITCHING:
            state = SwitchingState(count=0)
            return ReprLocSwitching(*args, **kwargs), state
        elif self is ReprLoc.UNIFORM:
            return _wrap_initloc(init_loc_uniform(*args, **kwargs)), None
        else:
            raise AssertionError("Unreachable")
