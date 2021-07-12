import dataclasses

from .core import Status


@dataclasses.dataclass
class AgeAndEnergy(Status):
    """
    Default implementation of agent's status.
    """

    age: int
    energy: float
    energy_delta: dataclasses.InitVar[float] = 0.01

    def __post_init__(self, energy_delta: float) -> None:
        # Make a constant function to ensure the delta is constant
        self._energy_delta = lambda: energy_delta

    def step(self) -> None:
        self.age += 1
        self.energy -= self._energy_delta()

    def update(self, *, energy_update: float) -> None:
        self.energy += energy_update
