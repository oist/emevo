from __future__ import annotations

from dataclasses import replace
from typing import Any

import chex
import jax
import jax.numpy as jnp

Self = Any


@chex.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: jax.Array
    energy: jax.Array
    is_alive: jax.Array
    capacity: float = 100.0
    metadata: dict[str, Any] | None = None

    def step(self) -> Self:
        """Get older."""
        return replace(self, age=self.age + 1)

    def update(self, *, energy_delta: jax.Array) -> Self:
        """Update energy."""
        energy = self.energy + jnp.where(
            self.is_alive,
            energy_delta,
            jnp.zeros_like(energy_delta),
        )
        return replace(self, energy=jnp.clip(energy, a_min=0.0, a_max=self.capacity))


def init_status(
    *,
    n: int,
    max_n: int,
    init_energy: float,
    capacity: float = 100.0,
    metadata: dict[str, Any] | None = None,
) -> Status:
    assert max_n >= n
    if max_n == n:
        is_alive = jnp.ones(n, dtype=bool)
    else:
        is_alive = jnp.concatenate(
            (jnp.ones(n, dtype=bool), jnp.zeros(max_n - n, dtype=bool))
        )
    return Status(
        age=jnp.zeros(max_n, dtype=jnp.int32),
        energy=jnp.ones(max_n, dtype=jnp.float32) * init_energy,
        is_alive=is_alive,
        capacity=capacity,
        metadata=metadata,
    )
