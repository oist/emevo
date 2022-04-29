"""
Abstract environment API.
These APIs define the environment and how an agent interacts with the environment.
Other specific things (e.g., asexual mating or sexual mating) are defiend in actual
environment implementations.
"""
import abc

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from emevo.body import Body, Encount
from emevo.types import Action, Info, Observation


class Environment(abc.ABC):
    """Abstract API for emevo environments"""

    @abc.abstractmethod
    def act(self, body: Body, action: Action) -> None:
        """An agent does a bodily action to the enviroment"""
        pass

    @abc.abstractmethod
    def available_bodies(self) -> Iterable[Body]:
        """Returns all bodies available in the environment"""
        pass

    @abc.abstractmethod
    def step(self) -> List[Encount]:
        """
        Steps the simulation one-step, according to the agents' actions.
        Returns all encounts.
        """
        pass

    @abc.abstractmethod
    def observe(self, body: Body) -> Optional[Tuple[Observation, Info]]:
        """Objective observation of environment"""
        pass

    @abc.abstractmethod
    def born(self, generation: int = 0, place: Optional[np.ndarray] = None) -> Body:
        """New agent is born in the enviroment. Return the body."""
        pass

    @abc.abstractmethod
    def die(self, body: Body) -> None:
        """Notify the environment that the body is dead."""
        pass

    @abc.abstractmethod
    def is_extinct(self) -> bool:
        """Return if agents are extinct"""
        pass

    def reset(self) -> None:
        """Do some initialization"""
        pass

    def close(self) -> None:
        """Close visualizer or so"""
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        """Set seed"""
        pass

    def render(self, mode: str) -> Union[None, np.ndarray]:
        """Render something to GUI or file"""
        pass
