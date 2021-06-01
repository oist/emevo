"""
Abstract environment APIs.
These APIs define the environment and how an agent interacts with the environment.
Other specific things (e.g., asexual mating or sexual mating) are defiend in actual
environment implementations.
"""
import abc
import dataclasses
import typing as t

import numpy as np


@dataclasses.dataclass()
class AgentBody:
    """A unique interface of an agent to interact with the environment.
    Or, the physical presence of the agent.
    Attributes:
    """

    actuator: t.Optional[t.Any]
    identifier: int
    is_dead: bool
    sensor: t.Optional[t.Any]

    def __deepcopy__(self) -> t.NoReturn:
        raise RuntimeError(
            "To ensure the uniqueness, deepcopy is not allowed for AgentBody."
        )


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
    parent: AgentBody

    def is_ready(self) -> bool:
        return self.parent.is_dead


@dataclasses.dataclass()
class Viviparous(Child):
    """A child stays in a parent's body for a while and will be born."""

    gene: np.ndarray
    parent: AgentBody
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0 or self.parent.is_dead

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


class MetricSpace(abc.ABC):
    """
    To represent physical contact of two robots, the  environment should consist
    of a metric space.
    In such a space, we can calculate a distance between two different elements.
    """

    @abc.abstractmethod
    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        pass


class EuclidSpace(MetricSpace):
    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        pass


class Environment(abc.ABC):
    @abc.abstractmethod
    def append_pending_action(self, body: AgentBody, action: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def execute_pending_actions(self) -> t.List[Child]:
        pass

    @abc.abstractmethod
    def give_observation(self, body: AgentBody) -> np.ndarray:
        pass

    @abc.abstractmethod
    def place_agent(
        self,
        body: AgentBody,
        position: t.Optional[t.Any] = None,
    ) -> None:
        pass


class _EnvironmentRegistory:
    """An internal class to register and make environments."""

    registered_envs: t.ClassVar[t.Dict[str, t.Type[Environment]]] = {}

    @classmethod
    def make(
        cls,
        env_class: t.Union[str, t.Type[Environment]],
        *args,
        **kwargs,
    ) -> Environment:
        if isinstance(env_class, str):
            env_class = cls.registered_envs.get(env_class.lower(), None)
        if not isinstance(env_class, type):
            raise ValueError(f"Invalid environmental class: {env_class}")
        return env_cls(*args, **kwargs)


def make(
    env_class: t.Union[str, t.Type[Environment]],
    *args,
    **kwargs,
) -> Environment:
    return _EnvironmentFactory.make(env_class, *args, **kwargs)


def register(name: str, env_class: t.Type[Environment]) -> None:
    _EnvironmentFactory.registered_envs[name] = env_class
