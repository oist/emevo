"""Abstract environment API"""
from __future__ import annotations

import abc
import dataclasses
from typing import Any, Generic, Protocol, TypeVar

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.spaces import Space
from emevo.status import Status
from emevo.types import Index
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
        return Profile(
            birthtime=self.birthtime.at[index].set(step),
            generation=self.generation.at[index].set(parent_gen + 1),
            unique_id=self.unique_id.at[index].set(uid),
        )

    def deactivate(self, index: Index) -> Self:
        return Profile(
            birthtime=self.birthtime.at[index].set(-1),
            generation=self.generation.at[index].set(-1),
            unique_id=self.unique_id.at[index].set(-1),
        )

    def is_active(self) -> jax.Array:
        return 0 <= self.unique_id


def init_profile(n: int, max_n: int) -> Profile:
    minus_1 = jnp.ones(max_n - n, dtype=jnp.int32) * -1
    return Profile(
        birthtime=jnp.concatenate((jnp.zeros(n, dtype=jnp.int32), minus_1)),
        generation=jnp.concatenate((jnp.zeros(n, dtype=jnp.int32), minus_1)),
        unique_id=jnp.concatenate((jnp.arange(n, dtype=jnp.int32), minus_1)),
    )


class ObsProtocol(Protocol):
    """Abstraction for agent's observation"""

    def as_array(self) -> jax.Array:
        ...


OBS = TypeVar("OBS", bound="ObsProtocol")


class StateProtocol(Protocol):
    """Environment's internal state"""

    key: chex.PRNGKey
    step: jax.Array
    profile: Profile
    status: Status
    n_born_agents: jax.Array

    def is_extinct(self) -> bool:
        ...


STATE = TypeVar("STATE", bound="StateProtocol")


@chex.dataclass
class TimeStep(Generic[OBS]):
    obs: OBS
    encount: jax.Array
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class Env(abc.ABC, Generic[STATE, OBS]):
    """Abstract API for emevo environments"""

    act_space: Space
    obs_space: Space

    def __init__(self, *args, **kwargs) -> None:
        # To supress PyRight errors in registry
        pass

    @abc.abstractmethod
    def reset(self, key: chex.PRNGKey) -> tuple[STATE, TimeStep[OBS]]:
        """Initialize environmental state."""
        pass

    @abc.abstractmethod
    def step(self, state: STATE, action: ArrayLike) -> tuple[STATE, TimeStep[OBS]]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def activate(
        self,
        state: STATE,
        parent_gen: jax.Array,
        init_energy: jax.Array,
    ) -> tuple[STATE, jax.Array]:
        """
        Mark an agent or some agents active.
        This method fails if there isn't enough space, returning (STATE, False).
        """
        pass

    @abc.abstractmethod
    def deactivate(self, state: STATE, index: Index) -> STATE:
        """
        Deactivate an agent or some agents. The shape of observations should remain the
        same so that `Env.step` is compiled onle once. So, to represent that an agent is
        dead, it is recommended to mark that body is not active and reuse it after a new
        agent is born.
        This method should not fail.
        """
        pass

    @abc.abstractmethod
    def visualizer(self, state: STATE, **kwargs) -> Visualizer:
        """Create a visualizer for the environment"""
        pass
