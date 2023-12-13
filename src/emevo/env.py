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
    capacity: float = 100.0

    def step(self) -> Self:
        """Get older."""
        return replace(self, age=self.age + 1)

    def activate(self, flag: jax.Array, init_energy: jax.Array) -> Self:
        age = jnp.where(flag, 0, self.age)
        energy = jnp.where(flag, init_energy, self.energy)
        return replace(self, age=age, energy=energy)

    def deactivate(self, flag: jax.Array) -> Self:
        return replace(self, age=jnp.where(flag, -1, self.age))

    def update(self, *, energy_delta: jax.Array) -> Self:
        """Update energy."""
        energy = self.energy + energy_delta
        return replace(self, energy=jnp.clip(energy, a_min=0.0, a_max=self.capacity))


def init_status(
    n: int,
    max_n: int,
    init_energy: float,
    capacity: float = 100.0,
) -> Status:
    assert max_n >= n
    return Status(
        age=jnp.zeros(max_n, dtype=jnp.int32),
        energy=jnp.ones(max_n, dtype=jnp.float32) * init_energy,
        capacity=capacity,
    )


@chex.dataclass
class Profile:
    """Agent profile."""

    birthtime: jax.Array
    generation: jax.Array
    unique_id: jax.Array

    def activate(self, flag: jax.Array, step: jax.Array) -> Self:
        birthtime = jnp.where(flag, step, self.birthtime)
        generation = jnp.where(flag, self.generation + 1, self.generation)
        unique_id = jnp.where(
            flag,
            jnp.cumsum(flag) + jnp.max(self.unique_id),
            self.unique_id,
        )
        return Profile(
            birthtime=birthtime,
            generation=generation,
            unique_id=unique_id,
        )

    def deactivate(self, flag: jax.Array) -> Self:
        return Profile(
            birthtime=jnp.where(flag, -1, self.birthtime),
            generation=jnp.where(flag, -1, self.generation),
            unique_id=jnp.where(flag, -1, self.unique_id),
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
