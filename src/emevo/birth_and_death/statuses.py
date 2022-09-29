"""Status, protocols, and implementations """

import dataclasses
from typing import Protocol


class HasAgeAndEnergy(Protocol):
    age: float
    energy: float


class HasEnergy(Protocol):
    energy: float


class Status(Protocol):
    def step(self) -> None:
        ...

    def update(self, **kwargs) -> None:
        ...


@dataclasses.dataclass
class AgeAndEnergy(Status, HasAgeAndEnergy):
    """Status with age and energy"""

    age: int
    energy: float
    energy_delta: dataclasses.InitVar[float] = 0.0

    def __post_init__(self, energy_delta: float) -> None:
        # Make a constant function to ensure the delta is constant
        self._energy_delta = lambda: energy_delta

    def step(self) -> None:
        self.age += 1
        self.energy -= self._energy_delta()

    def update(self, *, energy_update: float) -> None:
        self.energy += energy_update


@dataclasses.dataclass
class Energy(Status, HasEnergy):
    """
    Default implementation of agent's status.
    """

    energy: float
    energy_delta: dataclasses.InitVar[float] = 0.0

    def __post_init__(self, energy_delta: float) -> None:
        # Make a constant function to ensure the delta is constant
        self._energy_delta = lambda: energy_delta

    def step(self) -> None:
        self.energy -= self._energy_delta()

    def update(self, *, energy_update: float) -> None:
        self.energy += energy_update
