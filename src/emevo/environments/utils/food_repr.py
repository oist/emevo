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
    init_loc_choice,
    init_loc_uniform,
)

Self = Any


@chex.dataclass
class FoodNumState:
    current: jax.Array
    internal: jax.Array

    def appears(self) -> jax.Array:
        return (self.internal - self.current) >= 1.0

    def eaten(self, n: int | jax.Array) -> Self:
        return self.replace(current=self.current - n, internal=self.internal - n)

    def recover(self, n: int | jax.Array = 1) -> Self:
        return self.replace(current=self.current + n)


class ReprNumFn(Protocol):
    initial: int

    def __call__(self, state: FoodNumState) -> FoodNumState:
        ...


@dataclasses.dataclass(frozen=True)
class ReprNumConstant:
    initial: int

    def __call__(self, state: FoodNumState) -> FoodNumState:
        internal = jnp.fmax(state.current, state.internal)
        diff = jnp.clip(self.initial - state.current, a_min=0)
        state = state.replace(internal=internal + diff)
        return state


@dataclasses.dataclass(frozen=True)
class ReprNumLinear:
    initial: int
    dn_dt: float

    def __call__(self, state: FoodNumState) -> FoodNumState:
        # Increase the number of foods by dn_dt
        internal = jnp.fmax(state.current, state.internal)
        internal = jnp.clip(internal + self.dn_dt, a_max=float(self.initial))
        return state.replace(internal=internal)


@dataclasses.dataclass(frozen=True)
class ReprNumLogistic:
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, state: FoodNumState) -> FoodNumState:
        internal = jnp.fmax(state.current, state.internal)
        dn_dt = self.growth_rate * internal * (1 - internal / self.capacity)
        return state.replace(internal=internal + dn_dt)


class ReprNum(str, enum.Enum):
    """Methods to determine the number of foods reproduced."""

    CONSTANT = "constant"
    LINEAR = "linear"
    LOGISTIC = "logistic"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[ReprNumFn, FoodNumState]:
        if len(args) > 0:
            initial = args[0]
        elif "initial" in kwargs:
            initial = kwargs["initial"]
        else:
            raise ValueError("'initial' is required for all ReprNum functions")
        state = FoodNumState(  # type: ignore
            current=jnp.array(int(initial), dtype=jnp.int32),
            internal=jnp.array(float(initial), dtype=jnp.float32),
        )
        if self is ReprNum.CONSTANT:
            fn = ReprNumConstant(*args, **kwargs)
        elif self is ReprNum.LINEAR:
            fn = ReprNumLinear(*args, **kwargs)
        elif self is ReprNum.LOGISTIC:
            fn = ReprNumLogistic(*args, **kwargs)
        else:
            raise AssertionError("Unreachable")
        return fn, state


@chex.dataclass
class ReprLocState:
    n_produced: jax.Array

    def step(self) -> Self:
        return self.replace(n_produced=self.n_produced + 1)


ReprLocFn = Callable[[chex.PRNGKey, ReprLocState], jax.Array]


def _wrap_initloc(fn: InitLocFn) -> ReprLocFn:
    return lambda key, _: fn(key)


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

    def __call__(
        self,
        key: chex.PRNGKey,
        state: ReprLocState,
    ) -> jax.Array:
        count = state.n_produced + 1
        index = (count // self._interval) % self._n
        return jax.lax.switch(index, self._locfn_list, key)


class ReprLoc(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    CHOICE = "choice"
    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian-mixture"
    SWITCHING = "switching"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[ReprLocFn, Any]:
        state = ReprLocState(n_produced=jnp.array(0, dtype=jnp.int32))
        if self is ReprLoc.GAUSSIAN:
            return _wrap_initloc(init_loc_gaussian(*args, **kwargs)), state
        elif self is ReprLoc.GAUSSIAN_MIXTURE:
            return _wrap_initloc(init_loc_gaussian_mixture(*args, **kwargs)), state
        elif self is ReprLoc.CHOICE:
            return _wrap_initloc(init_loc_choice(*args, **kwargs)), state
        elif self is ReprLoc.SWITCHING:
            return ReprLocSwitching(*args, **kwargs), state
        elif self is ReprLoc.UNIFORM:
            return _wrap_initloc(init_loc_uniform(*args, **kwargs)), state
        else:
            raise AssertionError("Unreachable")
