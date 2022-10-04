"""Status implementations """

import dataclasses


@dataclasses.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: int
    energy: float

    def step(self) -> None:
        self.age += 1

    def update(self, *, energy_delta: float) -> None:
        energy = self.energy + energy_delta
        self.energy = max(0.0, energy)
