"""
Abstract environment API.
These APIs define the environment and how an agent interacts with the environment.
Other specific things (e.g., asexual mating or sexual mating) are defiend in actual
environment implementations.
"""
import abc
import dataclasses
import typing as t

import numpy as np

from emevo.body import Body
from emevo.types import Action, Observation, Rewards


@dataclasses.dataclass(frozen=True)
class Encount:
    """Two agents encount!"""

    bodies: t.Tuple[Body, Body]
    distance: float


class Environment(abc.ABC):
    """Abstract API for emevo environments"""

    @abc.abstractmethod
    def act(self, body: Body, action: Action) -> None:
        """An agent does a bodily action to the enviroment"""
        pass

    @abc.abstractmethod
    def available_bodies(self) -> t.Iterable[Body]:
        """Returns all bodies available in the environment"""
        pass

    @abc.abstractmethod
    def step(self) -> t.List[Encount]:
        """
        Steps the simulation one-step, according to the agents' actions.
        Returns all encounts.
        """
        pass

    @abc.abstractmethod
    def observe(self, body: Body) -> t.Optional[t.Tuple[Observation, Rewards]]:
        """Objective observation of environment"""
        pass

    @abc.abstractmethod
    def born(self, generation: int = 0, place: t.Optional[np.ndarray] = None) -> Body:
        pass

    @abc.abstractmethod
    def die(self, body: Body) -> None:
        pass

    def reset(self) -> None:
        """Do some initialization"""
        pass

    def close(self) -> None:
        """Close visualizer or so"""
        pass

    def seed(self, seed: t.Optional[int] = None) -> int:
        """Set seed"""
        pass

    def render(self, mode: str) -> t.Union[None, np.ndarray]:
        """Render something to GUI or file"""
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
