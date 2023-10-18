"""
Abstract environment API inspired by jumanji
"""
from __future__ import annotations

import abc
from typing import Any, Generic, Protocol, TypeVar

import chex
import jax
from jax.typing import ArrayLike

from emevo.types import Index, PyTree
from emevo.visualizer import Visualizer


@chex.dataclass
class Profile:
    """Agent profile."""

    birthtime: jax.Array
    generation: jax.Array
    index: jax.Array


class StateProtocol(Protocol):
    """Each state should have PRNG key"""

    key: chex.PRNGKey


STATE = TypeVar("STATE", bound="StateProtocol")

OBS = TypeVar("OBS")


@chex.dataclass
class TimeStep:
    encount: jax.Array | None
    obs: PyTree
    info: dict[str, Any]


class Env(abc.ABC, Generic[STATE, OBS]):
    """Abstract API for emevo environments"""

    def __init__(self, *args, **kwargs) -> None:
        # To supress PyRight errors in registry
        pass

    @abc.abstractmethod
    def reset(self, key: chex.PRNGKey) -> STATE:
        """Initialize environmental state."""
        pass

    @abc.abstractmethod
    def profile(self) -> Profile:
        """Returns profile of all 'alive' agents in the environment"""
        pass

    @abc.abstractmethod
    def step(self, state: STATE, action: ArrayLike) -> tuple[STATE, TimeStep]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def activate(self, state: STATE, index: Index) -> STATE:
        """Mark an agent or some agents active."""
        pass

    @abc.abstractmethod
    def deactivate(self, state: STATE, index: Index) -> STATE:
        """
        Deactivate an agent or some agents. The shape of observations should remain the
        same so that `Env.step` is compiled onle once. So, to represent that an agent is
        dead, it is recommended to mark that body is not active and reuse it after a new
        agent is born.
        """
        pass

    @abc.abstractmethod
    def is_extinct(self, state: STATE) -> bool:
        """Return if agents are extinct"""
        pass

    @abc.abstractmethod
    def visualizer(self, headless: bool = False, **kwargs) -> Visualizer:
        """Create a visualizer for the environment"""
        pass
