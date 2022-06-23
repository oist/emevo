"""
Abstract API for bodily existance of agents
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, NamedTuple, NoReturn
from uuid import uuid4

from numpy.typing import ArrayLike

from emevo.spaces import Space


class Locatable(abc.ABC):
    @abc.abstractmethod
    def location(self) -> ArrayLike:
        pass


@dataclasses.dataclass(frozen=True)
class Profile:
    """Unique id for an agent."""

    name: str
    generation: int
    birthtime: int | float

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError("Profile cannot be copied")


class Body(Locatable, abc.ABC):
    """
    Reprsents the bodily existance of the agent, also works as an effecient key object.
        return success

    def _try_placing_agent(self) -> NDArray | None:
        for _ in range(self._max_place_attempts):
            sampled = self._body_loc_fn(self._generator)
            if self._can_place(Vec2d(*sampled), self._agent_radius):
                return sampled
        return None

    def _try_placing_food(self, locations: list[Vec2d]) -> NDArray | None:
        for _ in range(self._max_place_attempts):
            sampled = self._food_loc_fn(self._generator, locations)
            if self._can_place(Vec2d(*sampled), self._food_radius):
                return sampled
        return None"""

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
