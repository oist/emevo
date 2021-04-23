""" Abstract environment API and some environment implementations.

This file is a standalone package and depends only on Python stdlib and numpy.
"""
import dataclasses
import enum

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, List, Type, Union

import numpy as np


MatingDelayFn = Callable[[np.random.RandomState], int]


class Space(ABC):
    """
    Represent state and action spaces.
    Simplified version of gym.Space, in that it can represent only numpy array.
    """

    dtype: np.dtype
    shape: np.shape

    @abstractmethod
    def clip(self, instance: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        pass


@dataclasses.dataclass()
class BoundedSpace(Space):
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    dtype: np.dtype = dataclasses.field(init=False)
    shape: np.shape = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.dtype = self.lower_bounds.dtype
        self.shape = self.upper_bounds.shape

    def clip(self, instance: np.ndarray) -> np.ndarray:
        return np.clip(instance, self.lower_bounds, self.upper_bounds)

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            dtype=self.dtype,
        )


@dataclasses.dataclass()
class UnboundedSpace(Space):
    dtype: np.dtype
    shape: np.shape

    def clip(self, instance: np.ndarray) -> np.ndarray:
        return instance

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.normal(dtype=self.dtype, shape=self.shape)


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
    def execute_pending_actions(self) -> List[MatingResult]:
        pass

    @abstractmethod
    def give_observation(self, agent_id: int) -> np.ndarray:
        pass

    @abstractmethod
    def place_agent(self, agend_id: int) -> None:
        pass


class Gridworld(Environment):
    def __init__(self) -> None:
        pass


class _EnvironmentFactory:
    """An internal class to register and make environments."""

    registered_envs: ClassVar[Dict[str, Type[Environment]]] = {"gridworld": Gridworld}

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


def make_environment(
    env_class: Union[str, Type[Environment]] = "gridworld",
    *args,
    **kwargs,
) -> Environment:
    return _EnvironmentFactory.make(env_class, *args, **kwargs)


def register_environment(name: str, env_class: Type[Environment]) -> None:
    _EnvironmentFactory.registered_envs[name] = env_class
