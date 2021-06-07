"""
Abstract environment API.
These APIs define the environment and how an agent interacts with the environment.
Other specific things (e.g., asexual mating or sexual mating) are defiend in actual
environment implementations.
"""
import abc
import typing as t

import numpy as np

from emevo.types import Observation


class Environment(abc.ABC):
    @abc.abstractmethod
    def append_pending_action(self, agent: Agent, action: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def execute_pending_actions(self) -> t.List[Child]:
        pass

    @abc.abstractmethod
    def observed_by(self, agent: Agent) -> np.ndarray:
        pass

    @abc.abstractmethod
    def place_agent(
        self,
        agent: Agent,
        position: t.Optional[t.Any] = None,
    ) -> Observation:
        pass

    def reset(self) -> None:
        """Want to do something before agents are placed?
        Put some funcitonalities here."""
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
