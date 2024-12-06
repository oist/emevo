from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from phyjax2d import Space as Physics
from phyjax2d import SpaceBuilder, Vec2d, make_approx_circle, make_square

from emevo.env import (
    Env,
    Status,
    TimeStep,
    UniqueID,
    Visualizer,
    init_status,
    init_uniqueid,
)
from emevo.environments.circle_foraging import (
    CFObs,
    CFState,
    CircleForaging,
    get_tactile,
    init_uniqueid,
    nstep,
)
from emevo.environments.env_utils import (
    CircleCoordinate,
    FoodNum,
    FoodNumFn,
    FoodNumState,
    Locating,
    LocatingFn,
    LocatingState,
    SquareCoordinate,
    first_to_nth_true,
    loc_gaussian,
    place,
    place_multi,
)
from emevo.spaces import BoxSpace, NamedTupleSpace

Self = Any


def _make_physics(
    dt: float,
    coordinate: CircleCoordinate | SquareCoordinate,
    linear_damping: float,
    angular_damping: float,
    n_velocity_iter: int,
    n_position_iter: int,
    n_max_agents: int,
    n_max_foods: int,
    agent_radius: float,
    food_radius: float,
    obstacles: Iterable[tuple[Vec2d, Vec2d]] = (),
) -> Physics:
    builder = SpaceBuilder(
        gravity=(0.0, 0.0),  # No gravity
        dt=dt,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        n_velocity_iter=n_velocity_iter,
        n_position_iter=n_position_iter,
        max_velocity=MAX_VELOCITY,
        max_angular_velocity=MAX_ANGULAR_VELOCITY,
    )
    # Set walls
    if isinstance(coordinate, CircleCoordinate):
        walls = make_approx_circle(coordinate.center, coordinate.radius)
    else:
        walls = make_square(
            *coordinate.xlim,
            *coordinate.ylim,
            rounded_offset=np.floor(food_radius * 2 / (np.sqrt(2) - 1.0)),
        )
    builder.add_chain_segments(chain_points=walls, friction=0.2, elasticity=0.4)
    for obs in obstacles:
        builder.add_segment(p1=obs[0], p2=obs[1], friction=0.2, elasticity=0.4)
    # Prepare agents
    for _ in range(n_max_agents):
        builder.add_circle(
            radius=agent_radius,
            friction=0.2,
            elasticity=0.4,
            density=0.1,
            color=AGENT_COLOR,
        )
    # Prepare foods
    for _ in range(n_max_foods):
        builder.add_circle(
            radius=food_radius,
            friction=0.2,
            elasticity=0.4,
            color=FOOD_COLOR,
            is_static=True,
        )
    return builder.build()


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


class CircleForagingWithPredator(CircleForaging):
    def __init__(
        self,
        *args,
        n_max_predator: int = 20,
        predator_radius: float = 20.0,
        n_predator_sensors: int = 20,
        n_predator_tactile_bins: int = 8,
        predator_sensor_length: int = 100.0,
        predator_max_force: float = 40.0,
        predator_min_force: float = -20.0,
        predator_init_energy: float = 20.0,
        predator_energy_capacity: float = 100.0,
        pradator_force_ec: float = 0.01 / 40.0,
        predator_basic_ec: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

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
