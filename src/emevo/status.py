from __future__ import annotations

from dataclasses import replace
from typing import Any

import chex
import jax
import jax.numpy as jnp

from emevo.types import Index

Self = Any


@chex.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: jax.Array
    energy: jax.Array
    capacity: float = 100.0

    def step(self) -> Self:
        """Get older."""
        return replace(self, age=self.age + 1)

    def activate(self, index: Index, init_energy: jax.Array) -> Self:
        age = self.age.at[index].set(0)
        energy = self.energy.at[index].set(init_energy)
        return replace(self, age=age, energy=energy)

    def deactivate(self, index: Index) -> Self:
        return replace(self, age=self.age.at[index].set(-1))

    def update(self, *, energy_delta: jax.Array) -> Self:
        """Update energy."""
        energy = self.energy + energy_delta
        return replace(self, energy=jnp.clip(energy, a_min=0.0, a_max=self.capacity))


def init_status(
    n: int,
    max_n: int,
    init_energy: float,
    capacity: float = 100.0,
) -> Status:
    assert max_n >= n
    return Status(
        age=jnp.zeros(max_n, dtype=jnp.int32),
        energy=jnp.ones(max_n, dtype=jnp.float32) * init_energy,
        capacity=capacity,
    )
