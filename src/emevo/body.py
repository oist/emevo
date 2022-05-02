"""
Abstract API for bodily existance of agents
"""

import abc
import dataclasses
import datetime as dt

from typing import Any, NoReturn, Tuple

import numpy as np

from gym import spaces


@dataclasses.dataclass(frozen=True)
class Profile:
    """Unique id for an agent."""

    name: str
    generation: int
    birthtime: dt.datetime = dataclasses.field(default_factory=dt.datetime.now)

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError("Profile cannot be copied")


class Body(abc.ABC):
    """
    Reprsents the bodily existance of the agent, also works as an effecient key object.
    """

    def __init__(self, name: str = "NoName", generation: int = 0, nth: int = 0) -> None:
        self.profile = Profile(name, generation)
        self.nth = nth

    @property
    @abc.abstractmethod
    def action_space(self) -> spaces.Space:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self) -> spaces.Space:
        pass

    @property
    @abc.abstractmethod
    def position(self) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        name, gen, birthtime = dataclasses.astuple(self.profile)
        birthtime = birthtime.strftime("%d,%h %H:%M:%S")
        return f"{name} (gen: {gen} birth: {birthtime})"

    def __eq__(self, other: Any) -> bool:
        return self.profile == other.profile

    def __ne__(self, other: Any) -> bool:
        return self.profile != other.profile

    def __hash__(self) -> int:
        return hash(self.profile)


@dataclasses.dataclass(frozen=True)
class Encount:
    """Two agents encounted each other"""

    bodies: Tuple[Body, Body]
    distance: float
