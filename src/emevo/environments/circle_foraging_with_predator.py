from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from phyjax2d import Circle, Color, Position, ShapeDict
from phyjax2d import Space as Physics
from phyjax2d import (
    SpaceBuilder,
    State,
    StateDict,
    Vec2d,
    circle_raycast,
    make_approx_circle,
    make_square_segments,
    segment_raycast,
)

from emevo.env import Status, TimeStep, init_uniqueid
from emevo.environments.circle_foraging import (
    AGENT_COLOR,
    FOOD_COLOR,
    MAX_ANGULAR_VELOCITY,
    MAX_VELOCITY,
    CFObs,
    CFState,
    CircleForaging,
    _first_n_true,
    _get_sensors,
    _nonzero,
    _SensorFn,
    get_tactile,
    nstep,
)
from emevo.environments.env_utils import CircleCoordinate, LocatingState

Self = Any
PREDATOR_COLOR: Color = Color(6, 214, 160)


def _observe_closest(
    shaped: ShapeDict,
    circle_prey: Circle,
    circle_predator: Circle,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
    state_prey: State,
    state_predator: State,
) -> jax.Array:
    rc = circle_raycast(0.0, 1.0, p1, p2, circle_prey, state_prey)
    to_prey = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = circle_raycast(0.0, 1.0, p1, p2, circle_predator, state_predator)
    to_predator = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = circle_raycast(0.0, 1.0, p1, p2, shaped.static_circle, stated.static_circle)
    to_sc = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = segment_raycast(1.0, p1, p2, shaped.segment, stated.segment)
    to_seg = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    obs = jnp.concatenate(
        jax.tree_util.tree_map(
            lambda arr: jnp.max(arr, keepdims=True),
            (to_prey, to_predator, to_sc, to_seg),
        ),
    )
    return jnp.where(obs == jnp.max(obs, axis=-1, keepdims=True), obs, -1.0)


_vmap_obs_closest = jax.vmap(
    _observe_closest,
    in_axes=(None, None, None, 0, 0, None, None, None),
)


def get_sensor_obs(
    shaped: ShapeDict,
    n_preys: int,
    n_sensors: int,
    sensor_range: tuple[float, float],
    sensor_length: float,
    predator_sensor_length: float,
    stated: StateDict,
) -> jax.Array:
    assert stated.circle is not None
    # Split shape and stated
    prey_shape, predator_shape = shaped.circle.split(n_preys)
    prey_state, predator_state = stated.circle.split(n_preys)
    p1_ag, p2_ag = _get_sensors(
        prey_shape,
        n_sensors,
        sensor_range,
        sensor_length,
        prey_state,
    )
    p1_pr, p2_pr = _get_sensors(
        predator_shape,
        n_sensors,
        sensor_range,
        predator_sensor_length,
        predator_state,
    )

    prey_obs = _vmap_obs_closest(
        shaped,
        prey_shape,
        predator_shape,
        p1_ag,
        p2_ag,
        stated,
        prey_state,
        predator_state,
    )
    predator_obs = _vmap_obs_closest(
        shaped,
        prey_shape,
        predator_shape,
        p1_pr,
        p2_pr,
        stated,
        prey_state,
        predator_state,
    )
    return jnp.concatenate((prey_obs, predator_obs), axis=0)


class _TactileInfo(NamedTuple):
    """
    N: Num. preys
    M: Num. predators
    B: Num. tactile bins
    F: Num. foods
    """

    prey2prey: jax.Array  # (N, N)
    predator2predator: jax.Array  # (M, M)
    tactile: jax.Array  # (N + M, 1, B)
    n_ate_food: jax.Array  # (N, 1)
    n_ate_prey: jax.Array  # (M, 1)
    eaten_foods: jax.Array  # (F,)
    eaten_preys: jax.Array  # (N,)


@chex.dataclass
class CFPredatorState(CFState[Status]):
    predator_loc: LocatingState
    n_born_predators: jax.Array


class CircleForagingWithPredator(CircleForaging):
    def __init__(
        self,
        n_max_predators: int = 20,
        predator_radius: float = 20.0,
        predator_sensor_length: int = 100,
        predator_init_energy: float = 20.0,
        predator_energy_capacity: float = 100.0,
        predator_force_ec: float = 0.01 / 40.0,
        predator_basic_ec: float = 0.0,
        **kwargs,
    ) -> None:
        self._n_max_predators = n_max_predators
        self._predator_radius = predator_radius
        self._predator_sensor_length = predator_sensor_length
        self._n_max_preys = kwargs["n_max_agents"] - n_max_predators
        assert self._n_max_preys > 0
        super().__init__(**kwargs)
        self._predator_init_energy = predator_init_energy
        self._predator_energy_capacity = predator_energy_capacity
        self._predator_force_ec = predator_force_ec
        self._predator_basic_ec = predator_basic_ec
        predator_act_ratio = (predator_radius**2) / (self._agent_radius**2)
        self._act_ratio = (
            jnp.ones((self.n_max_agents, 1))
            .at[self._n_max_preys :]
            .set(predator_act_ratio)
        )

    def _make_sensor_fn(self, observe_food_label: bool) -> _SensorFn:
        if observe_food_label:
            raise ValueError("Food label in predator env is not supported")
        else:
            return jax.jit(
                functools.partial(
                    get_sensor_obs,
                    n_preys=self._n_max_preys,
                    shaped=self._physics.shaped,
                    n_sensors=self._n_sensors,
                    predator_sensor_length=self._predator_sensor_length,
                    sensor_range=self._sensor_range_tuple,
                    sensor_length=self._sensor_length,
                )
            )

    def _make_physics(
        self,
        dt: float,
        linear_damping: float,
        angular_damping: float,
        n_velocity_iter: int,
        n_position_iter: int,
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
        if isinstance(self._coordinate, CircleCoordinate):
            walls = make_approx_circle(self._coordinate.center, self._coordinate.radius)
        else:
            walls = make_square_segments(
                *self._coordinate.xlim,
                *self._coordinate.ylim,
                rounded_offset=np.floor(self._food_radius * 2 / (np.sqrt(2) - 1.0)),
            )
        builder.add_chain_segments(chain_points=walls, friction=0.2, elasticity=0.4)
        for obs in obstacles:
            builder.add_segment(p1=obs[0], p2=obs[1], friction=0.2, elasticity=0.4)

        # Predators
        for _ in range(self._n_max_predators):
            builder.add_circle(
                radius=self._predator_radius,
                friction=0.2,
                elasticity=0.4,
                density=0.1,
                color=PREDATOR_COLOR,
            )
        # Agents
        for _ in range(self._n_max_preys):
            builder.add_circle(
                radius=self._agent_radius,
                friction=0.2,
                elasticity=0.4,
                density=0.1,
                color=AGENT_COLOR,
            )
        # Foods
        for _ in range(self._n_max_foods):
            builder.add_circle(
                radius=self._food_radius,
                friction=0.2,
                elasticity=0.4,
                color=FOOD_COLOR,
                is_static=True,
            )
        return builder.build()

    def step(  # type: ignore
        self,
        state: CFPredatorState,
        action: ArrayLike,
    ) -> tuple[CFPredatorState, TimeStep[CFObs]]:
        # Add force
        act = jax.vmap(self.act_space.clip)(jnp.array(action))
        f1_raw = jax.lax.slice_in_dim(act, 0, 1, axis=-1) * self._act_ratio
        f2_raw = jax.lax.slice_in_dim(act, 1, 2, axis=-1) * self._act_ratio
        f1 = jnp.concatenate((jnp.zeros_like(f1_raw), f1_raw), axis=1)
        f2 = jnp.concatenate((jnp.zeros_like(f2_raw), f2_raw), axis=1)
        circle = state.physics.circle
        circle = circle.apply_force_local(self._act_p1, f1)
        circle = circle.apply_force_local(self._act_p2, f2)
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
        # Get tactile obs
        tactile_info = self._collect_tactile(contacts, stated)
        # Gather sensor obs
        sensor_obs = self._sensor_obs(stated=stated)  # type: ignore
        force_norm = jnp.sqrt(f1_raw**2 + f2_raw**2).ravel()
        # energy_delta = food - coef * |force| for prey
        prey_energy_consumption = (
            self._force_energy_consumption * force_norm[: self._n_max_preys]
            + self._basic_energy_consumption
        )
        prey_energy_gain = jnp.sum(
            tactile_info.n_ate_food * self._food_energy_coef, axis=1
        )
        prey_energy_delta = prey_energy_gain - prey_energy_consumption
        # energy_delta = food - coef * |force| for predator
        predator_energy_consumption = (
            self._force_energy_consumption * force_norm[: self._n_max_preys]
            + self._basic_energy_consumption
        )
        predator_energy_gain = jnp.sum(
            tactile_info.n_ate_food * self._food_energy_coef, axis=1
        )
        predator_energy_delta = predator_energy_gain - predator_energy_consumption
        # Remove and regenerate foods
        key, food_key = jax.random.split(state.key)
        stated, food_num, food_loc, n_regen = self._remove_and_regenerate_foods(
            food_key,
            tactile_info.eaten_foods,  # (N_FOOD, N_LABEL)
            stated,
            state.step,
            state.food_num,
            state.food_loc,
        )
        status = state.status.update(
            energy_delta=jnp.concatenate(
                (prey_energy_delta, predator_energy_delta),
                axis=0,
            ),
            capacity=self._energy_capacity,
        )
        # Construct obs
        obs = CFObs(
            sensor=sensor_obs.reshape(-1, self._n_sensors, self._n_obj),
            collision=tactile_info.tactile,
            angle=stated.circle.p.angle,
            velocity=stated.circle.v.xy,
            angular_velocity=stated.circle.v.angle,
            energy=status.energy,
        )
        timestep = TimeStep(
            encount=[tactile_info.prey2prey, tactile_info.predator2predator],
            obs=obs,
            info={
                "energy_gain": jnp.concatenate(
                    (prey_energy_gain, predator_energy_gain),
                    axis=0,
                ),
                "energy_consumption": jnp.concatenate(
                    (prey_energy_consumption, predator_energy_consumption),
                    axis=0,
                ),
                "n_food_regenerated": n_regen,
                "n_food_eaten": jnp.sum(tactile_info.eaten_foods, axis=0),  # (N_LABEL,)
                "n_ate_food": jnp.concatenate(
                    (tactile_info.n_ate_food, tactile_info.n_ate_prey),
                    axis=0,
                ),
                "eaten_preys": tactile_info.eaten_preys,
            },
        )
        state = CFPredatorState(
            physics=stated,
            solver=solver,
            food_num=food_num,
            agent_loc=state.agent_loc,
            predator_loc=state.predator_loc,
            food_loc=food_loc,
            key=key,
            step=state.step + 1,
            unique_id=state.unique_id,
            status=status.step(state.unique_id.is_active()),
            n_born_agents=state.n_born_agents,
            n_born_predators=state.n_born_predators,
        )
        return state, timestep

    def _collect_tactile(self, contacts: jax.Array, stated: StateDict) -> _TactileInfo:
        c2c = self._physics.get_contact_mat("circle", "circle", contacts)
        c2sc = self._physics.get_contact_mat("circle", "static_circle", contacts)
        seg2c = self._physics.get_contact_mat("segment", "circle", contacts)
        prey_state, predator_state = stated.circle.split(self._n_max_preys)
        food_tactile, ft_raw = self._food_tactile(
            stated.static_circle.label,
            stated.circle,
            stated.static_circle,
            c2sc,
        )
        wall_tactile, _ = get_tactile(
            self._n_tactile_bins,
            stated.circle,
            stated.segment,
            seg2c.transpose(),
        )
        prey_prey_tactile, _ = get_tactile(
            self._n_tactile_bins,
            prey_state,
            prey_state,
            c2c[: self._n_max_preys, : self._n_max_preys],
        )
        prey_predator_tactile, _ = get_tactile(
            self._n_tactile_bins,
            prey_state,
            predator_state,
            c2c[: self._n_max_preys, self._n_max_preys :],
        )
        predator_prey_tactile, predator_prey_rawt = get_tactile(
            self._n_tactile_bins,
            predator_state,
            prey_state,
            c2c[self._n_max_preys :, : self._n_max_preys],
        )
        predator_predator_tactile, _ = get_tactile(
            self._n_tactile_bins,
            predator_state,
            predator_state,
            c2c[self._n_max_preys :, self._n_max_preys :],
        )
        self_tactile = jnp.concatenate(
            (prey_prey_tactile, predator_predator_tactile),
            axis=0,
        )
        other_tactile = jnp.concatenate(
            (prey_predator_tactile, predator_prey_tactile),
            axis=0,
        )
        tactile = jnp.concatenate(
            (self_tactile > 0, other_tactile > 0, food_tactile > 0, wall_tactile > 0),
            axis=1,
        )
        eaten_sum = jnp.sum(
            ft_raw[: self._n_max_preys, :, :, self._foraging_indices],
            axis=(0, 3),
        )
        eaten_preys_sum = jnp.sum(
            predator_prey_rawt[:, :, :, self._foraging_indices],
            axis=(0, 3),
        )
        return _TactileInfo(
            prey2prey=c2c[: self._n_max_preys, : self._n_max_preys],
            predator2predator=c2c[self._n_max_preys :, self._n_max_preys :],
            tactile=tactile,
            n_ate_food=jnp.sum(
                food_tactile[: self._n_max_preys, :, self._foraging_indices],
                axis=-1,
            ),
            n_ate_prey=jnp.sum(
                food_tactile[: self._n_max_preys, :, self._foraging_indices],
                axis=-1,
            ),
            eaten_foods=eaten_sum > 0,
            eaten_preys=eaten_preys_sum > 0,
        )

    def reset(self, key: chex.PRNGKey) -> tuple[CFState[Status], TimeStep[CFObs]]:
        prey_energy = jnp.ones(self._n_max_preys, dtype=jnp.float32) * self._init_energy
        predator_energy = (
            jnp.ones(self._n_max_predators, dtype=jnp.float32)
            * self._predator_init_energy
        )
        status = Status(
            age=jnp.zeros(self.n_max_agents, dtype=jnp.int32),
            energy=jnp.concatenate((prey_energy, predator_energy), axis=0),
        )
        physics, agent_loc, food_loc, food_num = self._initialize_physics_state(key)
        N = self.n_max_agents
        unique_id = init_uniqueid(self._n_initial_agents, N)
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
        sensor_obs = self._sensor_obs(stated=physics)  # type: ignore
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

    def activate(  # type: ignore
        self,
        state: CFPredatorState,
        is_parent: jax.Array,
    ) -> tuple[CFPredatorState, jax.Array]:
        N, M = self._n_max_preys, self._n_max_predators
        circle = state.physics.circle
        prey_circle, predator_circle = circle.split(self._n_max_preys)
        next_key, prey_key, predator_key = jax.random.split(state.key, 3)
        # Place prey
        prey_pos, prey_is_replaced, prey_parent_idx, prey_replaced_idx = self._place(
            N,
            state,
            prey_key,
            is_parent[: self._n_max_preys],
            prey_circle,
        )
        # Place predator
        (
            predator_pos,
            predator_is_replaced,
            predator_parent_idx,
            predator_replaced_idx,
        ) = self._place(
            M,
            state,
            predator_key,
            is_parent[self._n_max_preys :],
            predator_circle,
            offset=N,
        )
        is_replaced = jnp.concatenate((prey_is_replaced, predator_is_replaced))
        is_active = jnp.logical_or(is_replaced, circle.is_active)
        pos = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), prey_pos, predator_pos)
        physics = replace(
            state.physics,
            circle=replace(circle, p=pos, is_active=is_active),
        )
        unique_id = state.unique_id.activate(is_replaced)
        status = state.status.activate(
            self._energy_share_ratio,
            jnp.concatenate((prey_replaced_idx, predator_replaced_idx)),
            jnp.concatenate((prey_parent_idx, predator_parent_idx)),
        )
        n_children = jnp.sum(is_parent)
        new_state = replace(
            state,
            physics=physics,
            unique_id=unique_id,
            status=status,
            agent_loc=state.agent_loc.increment(n_children),
            n_born_agents=state.n_born_agents + n_children,
            key=next_key,
        )
        empty_id = jnp.ones_like(state.unique_id.unique_id) * -1
        unique_id_with_sentinel = jnp.concatenate(
            (state.unique_id.unique_id, jnp.zeros(1, dtype=jnp.int32))
        )
        replaced_indices = jnp.concatenate((prey_replaced_idx, predator_replaced_idx))
        parent_indices = jnp.concatenate((prey_parent_idx, predator_parent_idx))
        parent_id = empty_id.at[replaced_indices].set(
            unique_id_with_sentinel[parent_indices]
        )
        return new_state, parent_id

    def _place(
        self,
        n: int,
        state: CFPredatorState,
        key: chex.PRNGKey,
        is_parent: jax.Array,
        circle: State,
        offset: int = 0,
    ) -> tuple[Position, jax.Array, jax.Array, jax.Array]:
        keys = jax.random.split(key, n + 1)
        new_xy, ok = self._place_newborn(
            state.agent_loc,
            state.physics,
            keys[1:],
            circle.p.xy,
        )
        is_possible_parent = jnp.logical_and(
            is_parent,
            jnp.logical_and(circle.is_active, ok),
        )
        is_replaced = _first_n_true(
            jnp.logical_not(circle.is_active),
            jnp.sum(is_possible_parent),
        )
        is_parent = _first_n_true(is_possible_parent, jnp.sum(is_replaced))
        # parent_indices := nonzero_indices(parents) + (N, N, N, ....)
        parent_indices = _nonzero(is_parent, n) + offset
        # empty_indices := nonzero_indices(not(is_active)) + (N, N, N, ....)
        replaced_indices = _nonzero(is_replaced, n) + offset
        # To use .at[].add, append (0, 0) to sampled xy
        new_xy_with_sentinel = jnp.concatenate((new_xy, jnp.zeros((1, 2))))
        xy = circle.p.xy.at[replaced_indices].add(new_xy_with_sentinel[parent_indices])
        if self._random_angle:
            new_angle = jax.random.uniform(keys[0]) * jnp.pi * 2.0
            angle = jnp.where(is_replaced, new_angle, circle.p.angle)
        else:
            angle = jnp.where(is_replaced, 0.0, circle.p.angle)
        p = Position(angle=angle, xy=xy)
        return p, is_replaced, parent_indices, replaced_indices
