"""
Abstract environment APIs of emevo-gym.
"""
import dataclasses

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, NoReturn, Optional, Type, Union

import numpy as np


@dataclasses.dataclass()
class AgentBody:
    """A unique agent body.
    Attributes:
    """

    actuator: Optional[Any]
    identifier: int
    is_dead: bool
    sensor: Optional[Any]

    def __deepcopy__(self) -> NoReturn:
        raise RuntimeError(
            "To ensure the uniqueness, deepcopy is not allowed for AgentBody."
        )


class Child(ABC):
    """A class contains information of birth type."""

    gene: np.ndarray

    @abstractmethod
    def is_ready(self) -> bool:
        """Return if the child is ready to born or not."""
        pass

    def step(self) -> None:
        """Notify the child that the timestep has moved on."""
        pass


@dataclasses.dataclass()
class Oviparous(Child):
    """A child stays in an egg, then get a birth. """

    gene: np.ndarray
    positional_info: Any
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass()
class Virtual(Child):
    """Virtually replace a parent's mind, reusing the body. """

    gene: np.ndarray
    parent: AgentBody

    def is_ready(self) -> bool:
        return self.parent.is_dead


@dataclasses.dataclass()
class Viviparous(Child):
    """A child stays in a parent's body for a while, then get a birth. """

    gene: np.ndarray
    parent: AgentBody
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0 or self.parent.is_dead

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


class Environment(ABC):
    mating_type: ClassVar[MatingType]

    @abstractmethod
    def append_pending_action(self, body: AgentBody, action: np.ndarray) -> None:
        pass

    @abstractmethod
    def execute_pending_actions(self) -> List[Child]:
        pass

    @abstractmethod
    def give_observation(self, body: AgentBody) -> np.ndarray:
        pass

    @abstractmethod
    def place_agent(
        self,
        body: AgentBody,
        positional_info: Optional[Any] = None,
    ) -> None:
        pass


class _EnvironmentRegistory:
    """An internal class to register and make environments."""

    registered_envs: ClassVar[Dict[str, Type[Environment]]] = {}

    @classmethod
    def make(
        cls,
        env_class: Union[str, Type[Environment]],
        *args,
        **kwargs,
    ) -> Environment:
        if isinstance(env_class, str):
            env_class = cls.registered_envs.get(env_class.lower(), None)
        if not isinstance(env_class, type):
            raise ValueError(f"Invalid environmental class: {env_class}")
        return env_cls(*args, **kwargs)


def make(
    env_class: Union[str, Type[Environment]],
    *args,
    **kwargs,
) -> Environment:
    return _EnvironmentFactory.make(env_class, *args, **kwargs)


def register(name: str, env_class: Type[Environment]) -> None:
    _EnvironmentFactory.registered_envs[name] = env_class
