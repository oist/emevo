"""
Abstract environment API.
These APIs define the environment and how an agent interacts with the environment.
Other specific things (e.g., asexual mating or sexual mating) are defiend in actual
environment implementations.
"""
import abc
import dataclasses
import typing as t


from emevo.body import Body
from emevo.types import Action, Observation


@dataclasses.dataclass(frozen=True)
class Encount:
    bodies: t.Tuple[Body, Body]
    distance: float


@dataclasses.dataclass(frozen=True)
class Events:
    """
    Events that should be notified globally.
    """

    encounts: t.List[Encount]


@dataclasses.dataclass()
class Status:
    """
    Bodily status of an agent.
    Note that this is a 'default' implementation and downstreams environments
    should customize this for each usage.
    """

    energy: float
    injury: float
    lifetime: float


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
    def step(self) -> Events:
        """Steps the simulation one-step, according to the agents' actions."""
        pass

    @abc.abstractmethod
    def observe(self, body: Body) -> Observation:
        """Objective observation of environment"""
        pass

    @abc.abstractmethod
    def status(self, body: Body) -> Status:
        """Check a personal status"""
        pass

    @abc.abstractmethod
    def place(self, body: Body, place: np.ndarray) -> None:
        pass

    def reset(self) -> None:
        """Do some initialization"""
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
