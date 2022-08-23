"""
Abstract environment API.
"""
from __future__ import annotations

import abc
from typing import Generic, Protocol, TypeVar

from numpy.typing import NDArray

from emevo.body import LOC, Body, Encount
from emevo.visualizer import Visualizer


class Observation(Protocol):
    def __array__(self) -> NDArray:
        ...


ACT = TypeVar("ACT")
OBS = TypeVar("OBS", bound=Observation)


class Env(abc.ABC, Generic[ACT, LOC, OBS]):
    """Abstract API for emevo environments"""

    def __init__(self, *args, **kwargs) -> None:
        # To supress PyRight errors in registry
        pass

    @abc.abstractmethod
    def bodies(self) -> list[Body[LOC]]:
        """Returns all 'alive' bodies in the environment"""
        pass

    @abc.abstractmethod
    def step(self, actions: dict[Body[LOC], ACT]) -> list[Encount]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def observe(self, body: Body[LOC]) -> OBS:
        """Construct the observation from the state"""
        pass

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Do some initialization"""
        pass

    @abc.abstractmethod
    def born(self, location: LOC, generation: int) -> Body[LOC] | None:
        """Taken a location, generate and place a newborn in the environment."""
        pass

    @abc.abstractmethod
    def dead(self, body: Body[LOC]) -> None:
        """Remove a dead body from the environment."""
        pass

    @abc.abstractmethod
    def is_extinct(self) -> bool:
        """Return if agents are extinct"""
        pass

    @abc.abstractmethod
    def visualizer(self, *args, **kwargs) -> Visualizer:
        """Create a visualizer for the environment"""
        pass
