"""
Abstract API for bodily existance of agents
"""

from __future__ import annotations

import abc
import dataclasses
from collections import defaultdict
from typing import Any, Generic, NamedTuple, NoReturn, TypeVar

from emevo.spaces import Space

LOC = TypeVar("LOC")


class Locatable(abc.ABC, Generic[LOC]):
    @abc.abstractmethod
    def location(self) -> LOC:
        pass


@dataclasses.dataclass(frozen=True)
class Profile:
    """Unique id for an agent."""

    birthtime: int | float
    generation: int
    index: int


class Body(Locatable[LOC], abc.ABC):
    """
    Reprsents the bodily existance of the agent.
    Body should have an unique index, so it should work as an effecient key object.
    """

    _INDICES: dict[type, int] = defaultdict(int)

    def __init__(
        self,
        act_space: Space,
        obs_space: Space,
        generation: int = 0,
        birthtime: int | float = 0,
        index: int | None = None,
    ) -> None:
        self.act_space = act_space
        self.obs_space = obs_space
        if index is None:
            ty = type(self)
            index = self._INDICES[ty]
            self._INDICES[ty] += 1
        self._profile = Profile(birthtime, generation, index)

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError("Body cannot be copied")

    @property
    def generation(self) -> int:
        return self._profile.generation

    @property
    def index(self) -> int:
        return self._profile.index

    def info(self) -> Any:
        """Returns some information useful for debugging"""
        return None

    def __repr__(self) -> str:
        birthtime, gen, index = dataclasses.astuple(self._profile)
        return f"Body {index} (gen: {gen} birth: {birthtime})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Body):
            return self._profile == other._profile
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, Body):
            return self._profile != other._profile
        else:
            return True

    def __hash__(self) -> int:
        return hash(self._profile)


class Encount(NamedTuple):
    """Encounted two bodies"""

    a: Body
    b: Body
