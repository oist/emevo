from dataclasses import replace
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.env import Status, TimeStep
from emevo.environments.circle_foraging import (
    CFObs,
    CFState,
    CircleForaging,
    get_tactile,
    init_uniqueid,
    nstep,
)

Self = Any


@chex.dataclass
class StatusWithToxin(Status):
    toxin: jax.Array

    def deactivate(self, flag: jax.Array) -> Self:
        return replace(
            self,
            age=jnp.where(flag, 0, self.age),
            toxin=jnp.where(flag, 0, self.toxin),
        )


def init_status(max_n: int, init_energy: float) -> StatusWithToxin:
    return StatusWithToxin(
        age=jnp.zeros(max_n, dtype=jnp.int32),
        energy=jnp.ones(max_n, dtype=jnp.float32) * init_energy,
        toxin=jnp.zeros(max_n, dtype=jnp.float32),
    )


class CircleForagingWithNeurotoxin(CircleForaging):
    def __init__(
        self,
        *args,
        toxin_t0: float = 5.0,
        toxin_alpha: float = 1.0,
        toxin_delta: float = 10.0,
        toxin_recover_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._toxin_t0 = toxin_t0
        self._toxin_alpha = toxin_alpha
        self._toxin_recover_rate = toxin_recover_rate
        self._toxin_delta = toxin_delta
        assert self._n_food_sources - 1 == self._food_energy_coef.shape[1]

    def step(  # type: ignore
        self,
        state: CFState[StatusWithToxin],
        action: ArrayLike,
    ) -> tuple[CFState, TimeStep[CFObs]]:
        # Compute action decay ratio by toxin
        toxin_decay_rate = 1.0 / (
            1.0 + self._toxin_alpha * jnp.exp(self._toxin_t0 - state.status.toxin)
        )
        toxin_decay = jnp.expand_dims(1.0 - toxin_decay_rate, axis=1)
        # Add force
        act = jax.vmap(self.act_space.clip)(jnp.array(action))
        f1_raw = jax.lax.slice_in_dim(act, 0, 1, axis=-1)
        f2_raw = jax.lax.slice_in_dim(act, 1, 2, axis=-1)
        f1 = jnp.concatenate((jnp.zeros_like(f1_raw), f1_raw), axis=1)
        f2 = jnp.concatenate((jnp.zeros_like(f2_raw), f2_raw), axis=1)
        circle = state.physics.circle
        # Decay force by toxin
        circle = circle.apply_force_local(self._act_p1, f1 * toxin_decay)
        circle = circle.apply_force_local(self._act_p2, f2 * toxin_decay)
        stated = replace(state.physics, circle=circle)
        # Step physics simulator
        stated, solver, nstep_contacts = nstep(
            self._n_physics_iter,
            self._physics,
            stated,
            state.solver,
        )
        # Gather circle contacts
        contacts = jnp.max(nstep_contacts, axis=0)
        c2c = self._physics.get_contact_mat("circle", "circle", contacts)
        c2sc = self._physics.get_contact_mat("circle", "static_circle", contacts)
        seg2c = self._physics.get_contact_mat("segment", "circle", contacts)
        # Get tactile obs
        food_tactile, ft_raw = self._food_tactile(
            stated.static_circle.label,
            stated.circle,
            stated.static_circle,
            c2sc,
        )
        ag_tactile, _ = get_tactile(
            self._n_tactile_bins,
            stated.circle,
            stated.circle,
            c2c,
        )
        wall_tactile, _ = get_tactile(
            self._n_tactile_bins,
            stated.circle,
            stated.segment,
            seg2c.transpose(),
        )
        collision = jnp.concatenate(
            (ag_tactile > 0, food_tactile > 0, wall_tactile > 0),
            axis=1,
        )
        # Gather sensor obs
        sensor_obs = self._sensor_obs(stated=stated)
        # energy_delta = food - coef * |force|
        force_norm = jnp.sqrt(f1_raw**2 + f2_raw**2).ravel()
        energy_consumption = (
            self._force_energy_consumption * force_norm + self._basic_energy_consumption
        )
        n_ate = jnp.sum(food_tactile[:, :, self._foraging_indices], axis=-1)
        # toxin and foods
        n_ate_foods = n_ate[:, : self._n_food_sources - 1]  # (N-agents, N-foods)
        n_ate_toxin = n_ate[:, self._n_food_sources - 1]  # (N-agents,)
        energy_gain = jnp.sum(n_ate_foods * self._food_energy_coef, axis=1)
        energy_delta = energy_gain - energy_consumption
        # Remove and regenerate foods
        key, food_key = jax.random.split(state.key)
        eaten = jnp.sum(ft_raw[:, :, :, self._foraging_indices], axis=(0, 3)) > 0
        stated, food_num, food_loc, n_regen = self._remove_and_regenerate_foods(
            food_key,
            eaten,  # (N_FOOD, N_LABEL)
            stated,
            state.step,
            state.food_num,
            state.food_loc,
        )
        status = state.status.update(
            energy_delta=energy_delta,
            capacity=self._energy_capacity,
        )
        toxin = jnp.clip(
            status.toxin + n_ate_toxin * self._toxin_delta - self._toxin_recover_rate,
            min=0.0,
        )
        status = replace(status, toxin=toxin)
        # Construct obs
        obs = CFObs(
            sensor=sensor_obs.reshape(-1, self._n_sensors, self._n_obj),
            collision=collision,
            angle=stated.circle.p.angle,
            velocity=stated.circle.v.xy,
            angular_velocity=stated.circle.v.angle,
            energy=status.energy,
        )
        timestep = TimeStep(
            encount=c2c,
            obs=obs,
            info={
                "energy_gain": energy_gain,
                "energy_consumption": energy_consumption,
                "n_food_regenerated": n_regen,
                "n_food_eaten": jnp.sum(eaten, axis=0),  # (N_LABEL,)
                "n_ate_food": n_ate_foods,  # (N_AGENT, N_LABEL - 1)
                "n_ate_toxin": n_ate_toxin,  # (N_AGENT,)
            },
        )
        state = CFState(
            physics=stated,
            solver=solver,
            food_num=food_num,
            agent_loc=state.agent_loc,
            food_loc=food_loc,
            key=key,
            step=state.step + 1,
            unique_id=state.unique_id,
            status=status.step(state.unique_id.is_active()),
            n_born_agents=state.n_born_agents,
        )
        return state, timestep

    def reset(  # type: ignore
        self,
        key: chex.PRNGKey,
    ) -> tuple[CFState[StatusWithToxin], TimeStep[CFObs]]:
        physics, agent_loc, food_loc, food_num = self._initialize_physics_state(key)
        N = self.n_max_agents
        unique_id = init_uniqueid(self._n_initial_agents, N)
        status = init_status(N, self._init_energy)
        state = CFState(
            physics=physics,
            solver=self._physics.init_solver(),
            agent_loc=agent_loc,
            food_loc=food_loc,
            food_num=food_num,
            key=key,
            step=jnp.array(0, dtype=jnp.int32),
            unique_id=unique_id,
            status=status,
            n_born_agents=jnp.array(self._n_initial_agents, dtype=jnp.int32),
        )
        sensor_obs = self._sensor_obs(stated=physics)
        obs = CFObs(
            sensor=sensor_obs.reshape(-1, self._n_sensors, self._n_obj),
            collision=jnp.zeros((N, self._n_obj, self._n_tactile_bins), dtype=bool),
            angle=physics.circle.p.angle,
            velocity=physics.circle.v.xy,
            angular_velocity=physics.circle.v.angle,
            energy=state.status.energy,
        )
        # They shouldn't encount now
        timestep = TimeStep(encount=jnp.zeros((N, N), dtype=bool), obs=obs)
        return state, timestep
