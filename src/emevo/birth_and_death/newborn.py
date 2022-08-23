import abc
import dataclasses
from typing import Generic, Protocol

from emevo.body import Body
from emevo.env import LOC


class NewbornContext(Protocol[LOC]):
    generation: int
    location: LOC


@dataclasses.dataclass
class Newborn(abc.ABC, Generic[LOC]):
    """A class that contains information of birth type."""

    context: NewbornContext[LOC]

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return if the newborn is ready to be born or not."""
        pass

    def step(self) -> None:
        """Notify the newborn that the timestep has moved on."""
        pass


@dataclasses.dataclass
class Oviparous(Newborn):
    """A newborn stays in an egg for a while and will be born."""

    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Newborn.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass
class Viviparous(Newborn):
    """A newborn stays in a parent's body for a while and will be born."""

    parent: Body
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Newborn.step is called when it's ready")
        self.time_to_birth -= 1
