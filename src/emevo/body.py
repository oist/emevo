"""
Abstract API for bodily existance of agents
"""

import abc
import dataclasses
import datetime as dt
import typing as t
import uuid

from gym import spaces


@dataclasses.dataclass(frozen=True)
class Profile:
    """Unique id for an agent."""

    name: str
    generation: int
    birthtime: dt.datetime = dataclasses.field(default_factory=dt.datetime.now)
    uuid_: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)

    def __deepcopy__(self) -> t.NoReturn:
        raise RuntimeError("BodyID cannot be copied")

    def __eq__(self, other: t.Any) -> bool:
        return self.uuid_ == other.uuid_ and self.birthtime == other.birthtime

    def __ne__(self, other: t.Any) -> bool:
        return self.uuid_ != other.uuid_ or self.birthtime != other.birthtime

    def __hash__(self) -> bool:
        return hash(self.uuid_)


class Body(abc.ABC):
    """
    Reprsents the bodily existance of the agent, also works as an effecient key object.
    """

    def __init__(self, name: str = "NoName", generation: int = 0) -> None:
        self.profile = Profile(name, generation)

    @property
    @abc.abstractmethod
    def action_space(self) -> spaces.Space:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self) -> spaces.Space:
        pass

    def __repr__(self) -> str:
        name, gen, birthtime, uuid_ = dataclasses.astuple(self.profile)
        birthtime = birthtime.strftime("%d,%h %H:%M:%S")
        return f"{name} (gen: {gen} birth: {birthtime})"

    def __eq__(self, other: t.Any) -> bool:
        return self.profile == other.profile

    def __ne__(self, other: t.Any) -> bool:
        return self.profile != other.profile

    def __hash__(self) -> bool:
        return hash(self.profile)
