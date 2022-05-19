"""
Abstract API for bodily existance of agents
"""

import abc
import dataclasses

from typing import Any, NamedTuple, NoReturn, Tuple, Union

from numpy.typing import ArrayLike


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
        name: str = "NoName",
        generation: int = 0,
        birthtime: Union[int, float] = 0,
        index: int = 0,
    ) -> None:
        self._profile = Profile(name, generation, birthtime)
        self._index = index

    @abc.abstractmethod
    def act_shape(self) -> Tuple[int, ...]:
        pass

    @abc.abstractmethod
    def obs_shape(self) -> Tuple[int, ...]:
        pass

    def __repr__(self) -> str:
        name, gen, birthtime = dataclasses.astuple(self._profile)
        birthtime = birthtime.strftime("%d,%h %H:%M:%S")
        return f"{name} (gen: {gen} birth: {birthtime} index: {self._index})"

    def __eq__(self, other: Any) -> bool:
        return self._index == other.index

    def __ne__(self, other: Any) -> bool:
        return self._index == other.index


class Encount(NamedTuple):
    """Encounted two bodies"""

    a: Body
    b: Body
