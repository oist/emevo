from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple, overload

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.env import TimeStep
from emevo.environments.circle_foraging import CFObs, CFState, CircleForaging
from emevo.environments.phyjax2d import State
from emevo.spaces import BoxSpace, NamedTupleSpace


class CFSObs(NamedTuple):
    """Observation of an agent with smell."""

    sensor: jax.Array
    collision: jax.Array
    velocity: jax.Array
    angle: jax.Array
    angular_velocity: jax.Array
    energy: jax.Array
    smell: jax.Array
    smell_diff: jax.Array

    def as_array(self) -> jax.Array:
        return jnp.concatenate(
            (
                self.sensor.reshape(self.sensor.shape[0], -1),
                self.collision,
                self.velocity,
                jnp.expand_dims(self.angle, axis=1),
                jnp.expand_dims(self.angular_velocity, axis=1),
                jnp.expand_dims(self.energy, axis=1),
                self.smell,
                self.smell_diff,
            ),
            axis=1,
        )


def _as_cfsobs(obs: CFObs, smell: jax.Array, smell_diff: jax.Array) -> CFSObs:
    return CFSObs(
        sensor=obs.sensor,
        collision=obs.collision,
        angle=obs.angle,
        velocity=obs.velocity,
        angular_velocity=obs.angular_velocity,
        energy=obs.energy,
        smell=smell,
        smell_diff=smell_diff,
    )


@chex.dataclass
class CFSState(CFState):
    smell: jax.Array


def _as_cfsstate(state: CFState, smell: jax.Array) -> CFSState:
    return CFSState(
        physics=state.stated,
        solver=state.solver,
        food_num=state.food_num,
        agent_loc=state.agent_loc,
        food_loc=state.food_loc,
        key=state.key,
        step=state.step,
        unique_id=state.unique_id,
        status=state.status,
        n_born_agents=state.n_born_agents,
        smell=smell,
    )


def _compute_smell(
    n_food_sources: int,
    decay_factor: float,
    sc_state: State,
    sensor_xy: jax.Array,
) -> jax.Array:
    # Compute distance
    dist = jnp.linalg.norm(sc_state.p.xy - sensor_xy.reshape(1, 2), axis=1)
    smell = jnp.exp(-decay_factor * dist)
    smell_masked = jnp.where(sc_state.is_active, smell, 0.0)
    smell_per_source = jnp.zeros(n_food_sources).at[sc_state.label].add(smell_masked)
    return smell_per_source


_vmap_compute_smell = jax.vmap(_compute_smell, in_axes=(None, None, None, 0))


class CircleForagingWithSmell(CircleForaging):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        space = self.obs_space

        self.obs_space = NamedTupleSpace(
            CFSObs,
            sensor=space.spaces.sensor,  # type: ignore
            collision=space.spaces.collision,  # type: ignore
            velocity=space.spaces.velocity,  # type: ignore
            angle=space.spaces.angle,  # type: ignore
            angular_velocity=space.spaces.angular_velocity,  # type: ignore
            energy=space.spaces.energy,  # type: ignore
            smell=BoxSpace(
                low=0.0,
                high=float(self._n_max_foods),
                shape=(self._n_food_sources,),
            ),
            smell_diff=BoxSpace(
                low=-self._smell_diff_max,
                high=self._smell_diff_max,
                shape=(self._n_food_sources,),
            ),
        )

    def step(  # type: ignore
        self,
        state: CFSState,
        action: ArrayLike,
    ) -> tuple[CFSState, TimeStep[CFSObs]]:
        cf_state, ts = super().step(state, action)
        sensor_xy = cf_state.physics.circle.p.xy.at[:, 1].add(self._agent_radius)
        smell = _vmap_compute_smell(
            self._n_food_sources,
            self._smell_decay_factor,
            cf_state.physics.static_circle,
            sensor_xy,
        )
        smell_diff = jnp.clip(
            (smell - state.smell) * self._smell_diff_coef,
            a_min=-self._smell_diff_max,
            a_max=self._smell_diff_max,
        )
        state = _as_cfsstate(cf_state, smell)
        obs = _as_cfsobs(ts.obs, smell, smell_diff)
        return state, TimeStep(encount=ts.encount, obs=obs)

    def reset(  # type: ignore
        self,
        key: chex.PRNGKey,
    ) -> tuple[CFSState, TimeStep[CFSObs]]:
        cf_state, ts = super().reset(key)
        smell = _vmap_compute_smell(
            self._n_food_sources,
            self._smell_decay_factor,
            cf_state.physics.static_circle,
            cf_state.physics.circle.p.xy.at[:, 1].add(self._agent_radius),
        )
        state = _as_cfsstate(cf_state, smell)
        obs = _as_cfsobs(
            ts.obs,
            smell,
            jnp.zeros((self.n_max_agents, self._n_food_sources)),
        )
        return state, TimeStep(encount=ts.encount, obs=obs)

    def activate(  # type: ignore
        self,
        state: CFSState,
        is_parent: jax.Array,
    ) -> tuple[CFSState, jax.Array]:
        cf_state, parent_id = super().activate(state, is_parent)
        smell = _vmap_compute_smell(
            self._n_food_sources,
            self._smell_decay_factor,
            cf_state.physics.static_circle,
            cf_state.physics.circle.p.xy.at[:, 1].add(self._agent_radius),
        )
        new_state = _as_cfsstate(cf_state, smell)
        return new_state, parent_id
