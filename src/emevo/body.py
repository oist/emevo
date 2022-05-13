"""
Abstract API for bodily existance of agents
"""

import abc
import dataclasses

from typing import Any, NoReturn, Tuple

from emevo.types import Location, Shape


class Locatable(abc.ABC):
    @abc.abstractmethod
    def location(self) -> Location:
        pass


@dataclasses.dataclass(frozen=True)
class Profile:
    """Unique id for an agent."""

    name: str
    generation: int
    birthtime: float

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError("Profile cannot be copied")


class Body(Locatable, abc.ABC):
    """
    Reprsents the bodily existance of the agent, also works as an effecient key object.
    """

    def __init__(
        self,
        name: str = "NoName",
        generation: int = 0,
        birthtime: int = 0,
        index: int = 0,
    ) -> None:
        self._profile = Profile(name, generation, birthtime)
        self._index = index

    @abc.abstractmethod
    def act_shape(self) -> Shape:
        pass

    @abc.abstractmethod
    def obs_shape(self) -> Shape:
        pass

    def __repr__(self) -> str:
        name, gen, birthtime = dataclasses.astuple(self._profile)
        birthtime = birthtime.strftime("%d,%h %H:%M:%S")
        return f"{name} (gen: {gen} birth: {birthtime} index: {self._index})"

    def __eq__(self, other: Any) -> bool:
        return self._index == other.index

    def __ne__(self, other: Any) -> bool:
        return self._index == other.index


@dataclasses.dataclass(frozen=True)
class Encount:
    """Two agents encounted each other"""

    bodies: Tuple[Body, Body]
    distance: float
