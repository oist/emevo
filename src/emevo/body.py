"""
Abstract API for bodily existance of agents
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Generic, NamedTuple, NoReturn, TypeVar
from uuid import uuid4

from emevo.spaces import Space

LOC = TypeVar("LOC")


class Locatable(abc.ABC, Generic[LOC]):
    @abc.abstractmethod
    def location(self) -> LOC:
        pass


@dataclasses.dataclass(frozen=True)
class Profile:
    """Unique id for an agent."""

    name: str
    generation: int
    birthtime: int | float

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError("Profile cannot be copied")


class Body(Locatable[LOC], abc.ABC):
    """
    Reprsents the bodily existance of the agent, also works as an effecient key object.
    """

    def __init__(
        self,
        act_space: Space,
        obs_space: Space,
        name: str = "NoName",
        generation: int = 0,
        birthtime: int | float = 0,
    ) -> None:
        self.act_space = act_space
        self.obs_space = obs_space
        self.uuid = uuid4()
        self._profile = Profile(name, generation, birthtime)

    @property
    def generation(self) -> int:
        return self._profile.generation

    def info(self) -> Any:
        """Returns some information useful for debugging"""
        return None

    def __repr__(self) -> str:
        name, gen, birthtime = dataclasses.astuple(self._profile)
        return f"{name} (gen: {gen} birth: {birthtime} uuid: {self.uuid})"

    def __eq__(self, other: Any) -> bool:
        return self.uuid == other.uuid

    def __ne__(self, other: Any) -> bool:
        return self.uuid != other.uuid

    def __hash__(self) -> int:
        return hash(self.uuid)


class Encount(NamedTuple):
    """Encounted two bodies"""

    a: Body
    b: Body
