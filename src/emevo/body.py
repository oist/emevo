"""
Abstract API for bodily existance of agents
"""

import abc
import dataclasses

from typing import Any, NamedTuple, NoReturn, Union

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
    birthtime: Union[int, float]

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError("Profile cannot be copied")


class Body(Locatable, abc.ABC):
    """
    Reprsents the bodily existance of the agent, also works as an effecient key object.
    """

    def __init__(
        self,
        act_space: Space,
        obs_space: Space,
        name: str = "NoName",
        generation: int = 0,
        birthtime: Union[int, float] = 0,
        index: int = 0,
    ) -> None:
        self.act_space = act_space
        self.obs_space = obs_space
        self.index = index
        self._profile = Profile(name, generation, birthtime)

    def __repr__(self) -> str:
        name, gen, birthtime = dataclasses.astuple(self._profile)
        birthtime = birthtime.strftime("%d,%h %H:%M:%S")
        return f"{name} (gen: {gen} birth: {birthtime} index: {self.index})"

    def __eq__(self, other: Any) -> bool:
        return self.index == other.index

    def __ne__(self, other: Any) -> bool:
        return self.index == other.index

    def __hash__(self) -> int:
        return self.index


class Encount(NamedTuple):
    """Encounted two bodies"""

    a: Body
    b: Body
