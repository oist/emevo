import dataclasses


@dataclasses.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: float
    energy: float

    def step(self) -> None:
        """Get older."""
        self.age += 1

    def share(self, ratio: float) -> float:
        """Share some portion of energy."""
        shared = self.energy * ratio
        self.update(energy_delta=-shared)
        return shared

    def update(self, *, energy_delta: float) -> None:
        """Update energy."""
        energy = self.energy + energy_delta
        self.energy = max(0.0, energy)
