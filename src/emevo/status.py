from __future__ import annotations

import chex
from typing import Any

import jax
from emevo.types import Self
import jax.numpy as jnp

@chex.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: jax.Array
    energy: jax.Array
    capacity: float = 100.0
    metadata: dict[str, Any] | None = None

    def step(self) -> Self:
        """Get older."""
        return self.replace(age=self.age + 1)

    def share(self, ratio: float) -> tuple[Self, jax.Array]:
        """Share some portion of energy."""
        shared = self.energy * ratio
        return self.update(energy_delta=-shared), shared

    def update(self, *, energy_delta: jax.Array) -> Self:
        """Update energy."""
        energy = self.energy + energy_delta
        return self.replace(energy=jnp.clip(energy, a_min=0.0, a_max=self.capacity))
