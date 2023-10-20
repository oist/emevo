"""Abstract environment API"""
from __future__ import annotations

import abc
from typing import Any, Generic, Protocol, TypeVar

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.types import Index, PyTree
from emevo.visualizer import Visualizer

Self = Any


@chex.dataclass
class Profile:
    """Agent profile."""

    birthtime: jax.Array
    generation: jax.Array
    unique_id: jax.Array

    def activate(
        self,
        index: Index,
        parent_gen: jax.Array,
        uid: jax.Array,
        step: jax.Array,
    ) -> Self:
        unique_id = self.unique_id.at[index].set(uid)
        birthtime = self.birthtime.at[index].set(step)
        generation = self.generation.at[index].set(parent_gen + 1)
        return self.replace(
            birthtime=birthtime,
            generation=generation,
            unique_id=unique_id,
        )

    def deactivate(self, index: Index) -> Self:
        unique_id = self.unique_id.at[index].set(-1)
        birthtime = self.birthtime.at[index].set(-1)
        generation = self.generation.at[index].set(-1)
        return self.replace(
            birthtime=birthtime,
            generation=generation,
            unique_id=unique_id,
        )

    def is_active(self) -> jax.Array:
        return 0 <= self.unique_id


def init_profile(n: int, max_n: int) -> Profile:
    minus_1 = jnp.ones(max_n - n, dtype=jnp.int32) * -1
    birthtime = jnp.concatenate((jnp.zeros(n, dtype=jnp.int32), minus_1))
    generation = jnp.concatenate((jnp.zeros(n, dtype=jnp.int32), minus_1))
    unique_id = jnp.concatenate((jnp.arange(n, dtype=jnp.int32), minus_1))
    return Profile(
        birthtime=birthtime,
        generation=generation,
        unique_id=unique_id,
    )


class StateProtocol(Protocol):
    """Each state should have PRNG key"""

    key: chex.PRNGKey
    step: jax.Array
    profile: Profile
    n_born_agents: jax.Array


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
    def step(self, state: STATE, action: ArrayLike) -> tuple[STATE, TimeStep]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def activate(
        self,
        key: chex.PRNGKey,
        parent_gen: jax.Array,
        state: STATE,
    ) -> tuple[STATE, bool]:
        """Mark an agent or some agents active."""
        pass

    @abc.abstractmethod
    def deactivate(self, state: STATE) -> tuple[STATE, bool]:
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
