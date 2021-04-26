"""
Abstract environment API.
"""
import dataclasses
import enum

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Type, Union

import numpy as np


class BirthType(enum.Enum):
    EGG
    IMMEDIATE
    VIVIPARITY


class BirthInfo(ABC):
    """A class contains information of birth type."""

    pass


class Oviparous(BirthInfo):
    """A child stays in an egg, then get a birth. """

    positional_info: Any


class Virtual(BirthInfo):
    """Virtually replace a parent's mind, reusing the body. """

    pass


class Viviparous(BirthInfo):
    """A child stays in a parent's body for a while, then get a birth. """

    pass


class MatingType(enum.Enum):
    ASEXUAL = enum.auto
    SEXUAL = enum.auto


@dataclasses.dataclass()
class MatingConfig:
    mating_type: MatingType
    time_delay: MatingDelayFn


@dataclasses.dataclass
class Child:
    """
    A child who is waiting to get a birth.

    Attributes:
        birthinfo
        gene:  np.ndarray with dtype = np.uint8.
    """

    birth: Birth
    gene: np.ndarray
    parents: List[int]
    time_to_birth: int


class Environment(ABC):
    @abstractmethod
    def append_pending_action(self, action: np.ndarray) -> None:
        pass

    @abstractmethod
    def execute_pending_actions(self) -> List[Child]:
        pass

    @abstractmethod
    def give_observation(self, agent_id: int) -> np.ndarray:
        pass

    @abstractmethod
    def place_agent(self, agend_id: int) -> None:
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
