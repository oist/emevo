""" Utility functions to write food reproduction code in foraging environments.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Protocol

import chex
import jax
import jax.numpy as jnp

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
