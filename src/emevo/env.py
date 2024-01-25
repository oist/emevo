"""Abstract environment API"""
from __future__ import annotations

import abc
import dataclasses
from dataclasses import replace
from typing import Any, Generic, Protocol, TypeVar

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.spaces import Space
from emevo.visualizer import Visualizer

Self = Any


@chex.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: jax.Array
    energy: jax.Array

    def step(self) -> Self:
        """Get older."""
        return replace(self, age=self.age + 1)

    def activate(
        self,
        energy_share_ratio: float,
        child_indices: jax.Array,
        parent_indices: jax.Array,
    ) -> Self:
        age = self.age.at[child_indices].add(1)
        shared_energy = self.energy * energy_share_ratio
        shared_energy_with_sentinel = jnp.concatenate((shared_energy, jnp.zeros(1)))
        shared = shared_energy_with_sentinel[parent_indices]
        energy = self.energy.at[child_indices].set(shared)
        energy = energy.at[parent_indices].add(-shared)
        return replace(self, age=age, energy=energy)

    def deactivate(self, flag: jax.Array) -> Self:
        return replace(self, age=jnp.where(flag, -1, self.age))

    def update(self, energy_delta: jax.Array, capacity: float | None = 100.0) -> Self:
        """Update energy."""
        energy = self.energy + energy_delta
        return replace(self, energy=jnp.clip(energy, a_min=0.0, a_max=capacity))


def init_status(max_n: int, init_energy: float) -> Status:
    return Status(
        age=jnp.zeros(max_n, dtype=jnp.int32),
        energy=jnp.ones(max_n, dtype=jnp.float32) * init_energy,
    )


@chex.dataclass
class UniqueID:
    """Unique ID for agents. Starts from 1."""

    unique_id: jax.Array  # (N,)
    max_uid: jax.Array  # (1,)

    def activate(self, flag: jax.Array) -> Self:
        unique_id = jnp.where(
            flag,
            jnp.cumsum(flag) + self.max_uid,
            self.unique_id,
        )
        max_uid = self.max_uid + jnp.sum(flag)
        return UniqueID(unique_id=unique_id, max_uid=max_uid)

    def deactivate(self, flag: jax.Array) -> Self:
        return dataclasses.replace(self, unique_id=jnp.where(flag, -1, self.unique_id))

    def is_active(self) -> jax.Array:
        return 1 <= self.unique_id


def init_uniqueid(n: int, max_n: int) -> UniqueID:
    zeros = jnp.zeros(max_n - n, dtype=jnp.int32)
    return UniqueID(
        unique_id=jnp.concatenate((jnp.arange(1, n + 1, dtype=jnp.int32), zeros)),
        max_uid=jnp.array(max_n),
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
    unique_id: UniqueID
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
    n_max_agents: int

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
        is_parent: jax.Array,
    ) -> tuple[STATE, jax.Array]:
        """Mark some agents active, if possible."""
        pass

    @abc.abstractmethod
    def deactivate(self, state: STATE, flag: jax.Array) -> STATE:
        """
        Deactivate some agents. The shape of observations should remain the
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
