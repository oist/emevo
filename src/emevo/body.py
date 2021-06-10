"""
Abstract API for bodily existance of agents
"""

import abc
import dataclasses
import datetime as dt
import typing as t
import uuid

import numpy as np

from emevo.types import Action, Observation


@dataclasses.dataclass()
class BodyID:
    """Unique id for an agent."""

    generation: int
    name: str
    birthtime: dt.datetime = dataclasses.field(default_factory=dt.datetime.now)
    uuid_: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)

    def __deepcopy__(self) -> t.NoReturn:
        raise RuntimeError("BodyID cannot be copied")


class Body(abc.ABC):
    """Bodily existance of the agent"""

    def __init__(self, generation: int, name: str = "NoName") -> None:
        self.bodyid = BodyID(generation, name)

    @abc.abstractmethod
    def is_dead(self) -> bool:
        """Is this body dead?"""
        pass


class Agent(abc.ABC):
    agent_id: int
    is_dead: bool

    @abc.abstractmethod
    def __init__(self, agent_id: int, gene: np.ndarray, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def encode_gene(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def observe(self, prev_obs: Observation, action: Action, obs: Observation) -> None:
        """Observe the next state and returns."""
        pass

    @abc.abstractmethod
    def select_action(self, obs: Observation) -> Action:
        """Select an action based on an observation."""
        pass


class Child(abc.ABC):
    """A class contains information of birth type."""

    gene: np.ndarray

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return if the child is ready to be born or not."""
        pass

    def step(self) -> None:
        """Notify the child that the timestep has moved on."""
        pass


@dataclasses.dataclass()
class Oviparous(Child):
    """A child stays in an egg for a while and will be born."""

    gene: np.ndarray
    position: np.ndarray
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass()
class Virtual(Child):
    """Virtually replace a parent's mind, reusing the body."""

    gene: np.ndarray
    parent: Agent

    def is_ready(self) -> bool:
        return self.parent.is_dead


@dataclasses.dataclass()
class Viviparous(Child):
    """A child stays in a parent's body for a while and will be born."""

    gene: np.ndarray
    parent: Agent
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0 or self.parent.is_dead

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


class AgentManager:
    def __init__(self, agent_cls: t.Type[Agent]) -> None:
        self.agent_cls = agent_cls
        self.next_agent_id = 0

    def create_new_agent(self, gene: np.ndarray) -> Agent:
        """TODO: more information to pas"""
        pass

    def remove_dead_agents(self) -> t.List[int]:
        pass
