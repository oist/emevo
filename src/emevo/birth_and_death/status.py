from __future__ import annotations

import dataclasses
from typing import Any

from typing_extensions import Self


@dataclasses.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: float
    energy: float
    capacity: float = 100.0
    metadata: dict[str, Any] | None = None

    def step(self) -> None:
        """Get older."""
        self.age += 1

    def share(self, ratio: float) -> float:
        """Share some portion of energy."""
        shared = self.energy * ratio
        self.update(energy_delta=-shared)
        return shared

    def update(self, *, energy_delta: float) -> Self:
        """Update energy."""
        energy = self.energy + energy_delta
        self.energy = min(max(0.0, energy), self.capacity)
        return self
