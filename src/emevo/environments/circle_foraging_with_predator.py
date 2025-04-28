from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from phyjax2d import (
    Circle,
    Color,
    Position,
    ShapeDict,
)
from phyjax2d import Space as Physics
from phyjax2d import (
    SpaceBuilder,
    State,
    StateDict,
    Vec2d,
    circle_raycast,
    empty,
    make_approx_circle,
    make_square_segments,
    segment_raycast,
)

from emevo.env import Status, TimeStep, UniqueID
from emevo.environments.circle_foraging import (
    AGENT_COLOR,
    FOOD_COLOR,
    MAX_ANGULAR_VELOCITY,
    MAX_VELOCITY,
    NOWHERE,
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
from emevo.environments.env_utils import (
    CircleCoordinate,
    FoodNumState,
    LocatingState,
    LocGaussian,
    place,
)
from emevo.environments.smell import CFObsWithSmell, _vmap_compute_smell
from emevo.spaces import BoxSpace

Self = Any
PREDATOR_COLOR: Color = Color(135, 19, 21)


def _init_uniqueid(
    n: int | jax.Array,
    max_n: int,
    m: int | jax.Array,
    max_m: int,
) -> UniqueID:
    unique_id = jnp.concatenate(
        (
            jnp.arange(1, n + 1, dtype=jnp.int32),
            jnp.zeros(max_n - n, dtype=jnp.int32),
            jnp.arange(n + 1, n + m + 1, dtype=jnp.int32),
            jnp.zeros(max_m - m, dtype=jnp.int32),
        )
    )
    return UniqueID(
        unique_id=unique_id,
        max_uid=jnp.array(n + m + 1),
    )


def _observe_closest(
    shaped: ShapeDict,
    circle_prey: Circle,
    circle_predator: Circle,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
    state_prey: State,
    state_predator: State,
    ignore_sc: bool,
) -> jax.Array:
    rc = circle_raycast(0.0, 1.0, p1, p2, circle_prey, state_prey)
    to_prey = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = circle_raycast(0.0, 1.0, p1, p2, circle_predator, state_predator)
    to_predator = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = circle_raycast(0.0, 1.0, p1, p2, shaped.static_circle, stated.static_circle)
    to_sc = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    if ignore_sc:
        to_sc = jnp.ones_like(to_sc) * -1.0
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
    in_axes=(None, None, None, 0, 0, None, None, None, None),
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
        False,
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
        True,  # Predators ignore foods
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
    eaten_preys_per_predator: jax.Array  # (M, N)
    n_ate_food: jax.Array  # (N, 1)
    n_ate_prey: jax.Array  # (M, 1)
    eaten_foods: jax.Array  # (F,)
    eaten_preys: jax.Array  # (N,)


@chex.dataclass
class CFPredatorState(CFState[Status]):
    n_born_predators: jax.Array
    predator_eat_timer: jax.Array


class CircleForagingWithPredator(CircleForaging):
    def __init__(
        self,
        n_max_predators: int = 20,
        n_initial_predators: int = 10,
        predator_radius: float = 20.0,
        predator_sensor_length: int = 100,
        predator_init_energy: float = 20.0,
        predator_force_ec: float = 0.01 / 40.0,
        predator_basic_ec: float = 0.0,
        predator_digestive_rate: float = 0.9,
        predator_eat_interval: int = 10,
        predator_mouth_range: Literal["same", "narrow"] | list[int] = "same",
        **kwargs,
    ) -> None:
        self._n_max_predators = n_max_predators
        self._n_initial_predators = n_initial_predators
        self._predator_radius = predator_radius
        self._predator_sensor_length = predator_sensor_length
        self._n_max_preys = kwargs["n_max_agents"] - n_max_predators
        self._predator_eat_interval = predator_eat_interval
        assert self._n_max_preys > 0, f"Too many predators: {n_max_predators}"
        assert n_max_predators >= n_initial_predators, (
            f"Too many initial predators: {n_initial_predators}"
        )
        super().__init__(**kwargs, _n_additional_objs=1)

        if predator_mouth_range == "same":
            self._predator_foraging_indices = self._foraging_indices
        elif predator_mouth_range == "narrow":
            self._predator_foraging_indices = 0, self._n_tactile_bins - 1
        else:
            self._predator_foraging_indices = tuple(predator_mouth_range)

        self._predator_coordinate = self._coordinate
        self._predator_init_energy = predator_init_energy
        self._predator_force_ec = predator_force_ec
        self._predator_basic_ec = predator_basic_ec
        self._predator_digestive_rate = predator_digestive_rate
        predator_act_ratio = (predator_radius**2) / (self._agent_radius**2)
        self._act_ratio = (
            jnp.ones((self.n_max_agents, 1))
            .at[self._n_max_preys :]
            .set(predator_act_ratio)
        )
        shaped_nosc = replace(self._physics.shaped, static_circle=empty(Circle)())
        self._init_predator = jax.jit(
            functools.partial(
                place,
                n_trial=kwargs["max_place_attempts"],
                radius=self._predator_radius,
                coordinate=self._predator_coordinate,
                loc_fn=self._agent_loc_fn,
                shaped=shaped_nosc,
            )
        )
        if kwargs["newborn_loc"] == "uniform":

            def place_newborn_uniform(
                state: LocatingState,
                stated: StateDict,
                is_prey: bool,
                key: chex.PRNGKey,
                _: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return place(
                    n_trial=self._max_place_attempts,
                    radius=self._agent_radius if is_prey else self._predator_radius,
                    coordinate=(
                        self._coordinate if is_prey else self._predator_coordinate
                    ),
                    loc_fn=self._agent_loc_fn,
                    shaped=self._physics.shaped if is_prey else shaped_nosc,
                    loc_state=state,
                    key=key,
                    n_steps=0,
                    stated=stated,
                )

            self._place_newborn = jax.vmap(
                place_newborn_uniform,
                in_axes=(None, None, None, 0, None),
            )

        elif kwargs["newborn_loc"] == "neighbor":

            def place_newborn_neighbor(
                state: LocatingState,
                stated: StateDict,
                is_prey: bool,
                key: chex.PRNGKey,
                agent_loc: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                loc_fn = LocGaussian(
                    agent_loc,
                    jnp.ones_like(agent_loc) * kwargs["neighbor_stddev"],
                )

                return place(
                    n_trial=self._max_place_attempts,
                    radius=self._agent_radius if is_prey else self._predator_radius,
                    coordinate=(
                        self._coordinate if is_prey else self._predator_coordinate
                    ),
                    loc_fn=loc_fn,
                    shaped=self._physics.shaped if is_prey else shaped_nosc,
                    loc_state=state,
                    key=key,
                    n_steps=0,
                    stated=stated,
                )

            self._place_newborn = jax.vmap(
                place_newborn_neighbor,
                in_axes=(None, None, None, 0, 0),
            )
        else:
            raise ValueError("Invalid newborn_loc")

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
        # Set obstacles first
        for obs in obstacles:
            builder.add_segment(p1=obs[0], p2=obs[1], friction=0.2, elasticity=0.4)
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

        # Preys
        for _ in range(self._n_max_preys):
            builder.add_circle(
                radius=self._agent_radius,
                friction=0.2,
                elasticity=0.4,
                density=0.1,
                color=AGENT_COLOR,
            )
        # Predators
        for _ in range(self._n_max_predators):
            builder.add_circle(
                radius=self._predator_radius,
                friction=0.2,
                elasticity=0.4,
                density=0.1,
                ignore=["static_circle"],
                color=PREDATOR_COLOR,
            )
        print(self._n_max_preys, self._n_max_predators)
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
        tactile_info = self._collect_tactile(
            contacts,
            stated,
            state.predator_eat_timer <= 0,
        )
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
            self._predator_force_ec * force_norm[self._n_max_preys :]
            + self._predator_basic_ec
        )
        prey_energies = state.status.energy[: self._n_max_preys]
        predator_energy_gain = self._predator_digestive_rate * jnp.matmul(
            tactile_info.eaten_preys_per_predator,
            prey_energies,
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
                "eaten_preys": jnp.ravel(tactile_info.eaten_preys),
            },
        )
        predator_eat_timer = jnp.where(
            predator_energy_gain > 0,
            self._predator_eat_interval,
            state.predator_eat_timer - 1,
        )
        state = CFPredatorState(
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
            n_born_predators=state.n_born_predators,
            predator_eat_timer=predator_eat_timer,
        )
        return state, timestep

    def _collect_tactile(
        self,
        contacts: jax.Array,
        stated: StateDict,
        can_eat: jax.Array,
    ) -> _TactileInfo:
        c2c = self._physics.get_contact_mat("circle", "circle", contacts)
        c2sc = self._physics.get_contact_mat("circle", "static_circle", contacts)
        seg2c = self._physics.get_contact_mat("segment", "circle", contacts)
        prey_state, predator_state = stated.circle.split(self._n_max_preys)
        food_tactile, ft_raw = self._food_tactile(
            stated.static_circle.label,
            prey_state,
            stated.static_circle,
            c2sc,
        )
        wall_tactile, _ = get_tactile(
            self._n_tactile_bins,
            stated.circle,
            stated.segment,
            seg2c.transpose(),
            shift=self._tactile_shift,
        )
        prey_prey_tactile, _ = get_tactile(
            self._n_tactile_bins,
            prey_state,
            prey_state,
            c2c[: self._n_max_preys, : self._n_max_preys],
            shift=self._tactile_shift,
        )
        prey_predator_tactile, _ = get_tactile(
            self._n_tactile_bins,
            prey_state,
            predator_state,
            c2c[: self._n_max_preys, self._n_max_preys :],
            shift=self._tactile_shift,
        )
        predator_prey_tactile, predator_prey_rawt = get_tactile(
            self._n_tactile_bins,
            predator_state,
            prey_state,
            c2c[self._n_max_preys :, : self._n_max_preys],
            shift=self._tactile_shift,
        )
        predator_predator_tactile, _ = get_tactile(
            self._n_tactile_bins,
            predator_state,
            predator_state,
            c2c[self._n_max_preys :, self._n_max_preys :],
            shift=self._tactile_shift,
        )
        self_tactile = jnp.concatenate(
            (prey_prey_tactile, predator_predator_tactile),
            axis=0,
        )
        other_tactile = jnp.concatenate(
            (prey_predator_tactile, predator_prey_tactile),
            axis=0,
        )
        # Extend zero vector to food tactile
        food_tactile_extended = jnp.concatenate(
            (
                food_tactile > 0,
                jnp.zeros((self._n_max_predators, 1, self._n_tactile_bins), dtype=bool),
            )
        )
        # (food_tactile > 0).at[self._n_max_preys :].set(False)
        tactile = jnp.concatenate(
            (
                self_tactile > 0,
                other_tactile > 0,
                food_tactile_extended,
                wall_tactile > 0,
            ),
            axis=1,
        )
        eaten_preys_per_predator = jnp.where(
            can_eat.reshape(self._n_max_predators, 1, 1, 1),
            predator_prey_rawt[:, :, :, self._predator_foraging_indices],
            False,
        )
        return _TactileInfo(
            prey2prey=c2c[: self._n_max_preys, : self._n_max_preys],
            predator2predator=c2c[self._n_max_preys :, self._n_max_preys :],
            tactile=tactile,
            eaten_preys_per_predator=jnp.squeeze(
                jnp.max(eaten_preys_per_predator, axis=-1),
                axis=-1,
            ),
            n_ate_food=jnp.sum(
                food_tactile[: self._n_max_preys, :, self._foraging_indices],
                axis=-1,
            ),
            n_ate_prey=jnp.sum(
                eaten_preys_per_predator,
                axis=(1, 3),
            ),
            eaten_foods=jnp.max(
                ft_raw[: self._n_max_preys, :, :, self._foraging_indices],
                axis=(0, 3),
            ),
            eaten_preys=jnp.max(eaten_preys_per_predator, axis=(0, 3)),
        )

    def reset(self, key: chex.PRNGKey) -> tuple[CFPredatorState, TimeStep[CFObs]]:
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
        is_active = physics.circle.is_active
        n_preys = jnp.sum(is_active[: self._n_max_preys])
        n_predators = jnp.sum(is_active[self._n_max_preys :])
        unique_id = _init_uniqueid(
            n_preys,
            self._n_max_preys,
            n_predators,
            self._n_max_predators,
        )
        state = CFPredatorState(
            physics=physics,
            solver=self._physics.init_solver(),
            agent_loc=agent_loc,
            food_loc=food_loc,
            food_num=food_num,
            key=key,
            step=jnp.array(0, dtype=jnp.int32),
            unique_id=unique_id,
            status=status,
            n_born_agents=n_preys,
            n_born_predators=n_predators,
            predator_eat_timer=jnp.zeros(self._n_max_predators, dtype=jnp.int32),
        )
        sensor_obs = self._sensor_obs(stated=physics)  # type: ignore
        N = self.n_max_agents
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
            True,
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
            False,
            state,
            predator_key,
            is_parent[self._n_max_preys :],
            predator_circle,
        )
        is_replaced = jnp.concatenate((prey_is_replaced, predator_is_replaced))
        is_active = jnp.logical_or(is_replaced, circle.is_active)
        pos = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), prey_pos, predator_pos)
        physics = replace(
            state.physics,
            circle=replace(circle, p=pos, is_active=is_active),
        )
        unique_id = state.unique_id.activate(is_replaced)
        replaced_indices = jnp.concatenate(
            # Replace the sentinel to n_max_agents
            (
                jnp.where(
                    prey_replaced_idx == self._n_max_preys,
                    self.n_max_agents,
                    prey_replaced_idx,
                ),
                predator_replaced_idx + self._n_max_preys,
            )
        )
        parent_indices = jnp.concatenate(
            (
                jnp.where(
                    prey_parent_idx == self._n_max_preys,
                    self.n_max_agents,
                    prey_parent_idx,
                ),
                predator_parent_idx + self._n_max_preys,
            )
        )
        status = state.status.activate(
            self._energy_share_ratio,
            replaced_indices,
            parent_indices,
        )
        n_children = jnp.sum(is_replaced)
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
        parent_id = empty_id.at[replaced_indices].set(
            unique_id_with_sentinel[parent_indices]
        )
        return new_state, parent_id

    def _place(
        self,
        n: int,
        is_prey: bool,
        state: CFPredatorState,
        key: chex.PRNGKey,
        is_parent: jax.Array,
        circle: State,
    ) -> tuple[Position, jax.Array, jax.Array, jax.Array]:
        keys = jax.random.split(key, n + 1)
        new_xy, ok = self._place_newborn(
            state.agent_loc,
            state.physics,
            is_prey,
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
        parent_indices = _nonzero(is_parent, n)
        # replaced_indices := nonzero_indices(not(is_active)) + (N, N, N, ....)
        replaced_indices = _nonzero(is_replaced, n)
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

    def _initialize_physics_state(
        self,
        key: chex.PRNGKey,
    ) -> tuple[StateDict, LocatingState, list[LocatingState], list[FoodNumState]]:
        # Set segment
        stated = self._physics.shaped.zeros_state()
        assert stated.circle is not None

        # Move all circle to the invisiable area
        stated = stated.nested_replace(
            "circle.p.xy",
            jnp.ones_like(stated.circle.p.xy) * NOWHERE,
        )
        # Preys have label 0, and predators have label1
        stated = stated.nested_replace(
            "circle.label",
            stated.circle.label.at[self._n_max_preys :].set(1),
        )
        stated = stated.nested_replace(
            "static_circle.p.xy",
            jnp.ones_like(stated.static_circle.p.xy) * NOWHERE,
        )

        key, *agent_keys = jax.random.split(key, self._n_initial_agents + 1)
        agentloc_state = self._initial_agentloc_state
        is_active_preys = []
        for i, key in enumerate(agent_keys):
            xy, ok = self._init_agent(
                loc_state=agentloc_state,
                key=key,
                n_steps=i,
                stated=stated,
            )
            if ok:
                stated = stated.nested_replace(
                    "circle.p.xy",
                    stated.circle.p.xy.at[i].set(xy),
                )
                agentloc_state = agentloc_state.increment()
            is_active_preys.append(ok)

        n_preys = sum(is_active_preys)
        if n_preys < self._n_initial_agents:
            diff = self._n_initial_agents - n_preys
            warnings.warn(f"Failed to place {diff} preys!", stacklevel=1)

        key, *predator_keys = jax.random.split(key, self._n_initial_predators + 1)
        is_active_predators = []
        for i, key in enumerate(predator_keys):
            xy, ok = self._init_predator(
                loc_state=agentloc_state,
                key=key,
                n_steps=i,
                stated=stated,
            )
            index = i + self._n_max_preys
            if ok:
                stated = stated.nested_replace(
                    "circle.p.xy",
                    stated.circle.p.xy.at[index].set(xy),
                )
                agentloc_state = agentloc_state.increment()
            is_active_predators.append(ok)

        n_predators = sum(is_active_predators)
        if n_predators < self._n_initial_predators:
            diff = self._n_initial_predators - n_predators
            warnings.warn(f"Failed to place {diff} predators", stacklevel=1)

        # Set is_active
        is_active_c = jnp.concatenate(
            (
                jnp.array(is_active_preys),
                jnp.zeros(self._n_max_preys - self._n_initial_agents, dtype=bool),
                jnp.array(is_active_predators),
                jnp.zeros(
                    self._n_max_predators - self._n_initial_predators,
                    dtype=bool,
                ),
            )
        )
        # Fill 0 for food
        is_active_s = jnp.zeros(self._n_max_foods, dtype=bool)
        stated = stated.nested_replace("circle.is_active", is_active_c)
        stated = stated.nested_replace("static_circle.is_active", is_active_s)

        if self._random_angle:
            key, angle_key = jax.random.split(key)
            angle = jax.random.uniform(
                angle_key,
                shape=stated.circle.p.angle.shape,
                maxval=2.0 * jnp.pi,
            )
            stated = stated.nested_replace("circle.p.angle", angle)

        food_failed = 0
        foodloc_states = [s for s in self._initial_foodloc_states]
        foodnum_states = [s for s in self._initial_foodnum_states]
        foodkeys = jax.random.split(key, self._n_food_sources)
        del key
        for i, foodkey in enumerate(foodkeys):
            n_initial = self._food_num_fns[i].initial
            xy, ok = self._place_food_fns[i](
                loc_state=foodloc_states[i],
                n_max_placement=n_initial,
                key=foodkey,
                n_steps=i,
                stated=stated,
            )
            n = jnp.sum(ok)
            is_active = stated.static_circle.is_active
            place = jax.jit(_first_n_true)(jnp.logical_not(is_active), n)
            stated = stated.nested_replace(
                "static_circle.p.xy",
                stated.static_circle.p.xy.at[place].set(xy[ok]),
            )
            stated = stated.nested_replace(
                "static_circle.is_active",
                jnp.logical_or(place, is_active),
            )
            # Set food label
            stated = stated.nested_replace(
                "static_circle.label",
                stated.static_circle.label.at[place].set(i),
            )
            # Set is_active
            foodloc_states[i] = foodloc_states[i].increment(n)
            foodnum_states[i] = foodnum_states[i].recover(n)
            food_failed += n_initial - n

        if food_failed > 0:
            warnings.warn(f"Failed to place {food_failed} foods!", stacklevel=1)

        return stated, agentloc_state, foodloc_states, foodnum_states


def _mask_self(smell_arr: jax.Array) -> jax.Array:
    identity_matrix = jnp.eye(smell_arr.shape[0])
    return jnp.where(identity_matrix, 0.0, smell_arr)


class CFPredatorWithSmell(CircleForagingWithPredator):
    def __init__(
        self,
        *args,
        smell_decay_factor: float = 0.1,
        smell_front_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        old_obs_space = self.obs_space
        self.obs_space = old_obs_space.extend(  # type: ignore
            CFObsWithSmell,
            smell=BoxSpace(low=0.0, high=1.0, shape=(2,)),
        )
        self._smell_decay_factor = smell_decay_factor
        self._smell_front_only = smell_front_only
        to_prey_nose = jnp.tile(
            jnp.array([[0.0, self._agent_radius]]),
            (self._n_max_preys, 1),
        )
        to_predator_nose = jnp.tile(
            jnp.array([[0.0, self._predator_radius]]),
            (self._n_max_predators, 1),
        )
        self._to_nose = jnp.concatenate(
            (to_prey_nose, to_predator_nose),
            axis=0,
        )

    def _smell(self, circle: State) -> jax.Array:
        center = circle.p.xy
        nose = circle.p.xy + jnp.array([[0.0, self._agent_radius]])
        rotated_nose = circle.p.rotate(nose)
        smell = _vmap_compute_smell(  # N_preys, N_total
            self._smell_decay_factor,
            self._smell_front_only,
            circle,
            center,
            rotated_nose,
        )
        masked_smell = _mask_self(smell)
        to_prey = jnp.clip(  # N_total,
            jnp.sum(masked_smell[:, : self._n_max_preys], axis=1, keepdims=True),
            max=1.0,
        )
        to_predator = jnp.clip(
            jnp.sum(masked_smell[:, self._n_max_preys :], axis=1, keepdims=True),
            max=1.0,
        )
        return jnp.concatenate((to_prey, to_predator), axis=1)

    def step(  # type: ignore
        self,
        state: CFPredatorState,
        action: ArrayLike,
    ) -> tuple[CFPredatorState, TimeStep[CFObsWithSmell]]:
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
        tactile_info = self._collect_tactile(
            contacts,
            stated,
            state.predator_eat_timer <= 0,
        )
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
            self._predator_force_ec * force_norm[self._n_max_preys :]
            + self._predator_basic_ec
        )
        prey_energies = state.status.energy[: self._n_max_preys]
        predator_energy_gain = self._predator_digestive_rate * jnp.matmul(
            tactile_info.eaten_preys_per_predator,
            prey_energies,
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
        # Smell
        smell = self._smell(stated.circle)
        # Construct obs
        obs = CFObsWithSmell(
            sensor=sensor_obs.reshape(-1, self._n_sensors, self._n_obj),
            collision=tactile_info.tactile,
            angle=stated.circle.p.angle,
            velocity=stated.circle.v.xy,
            angular_velocity=stated.circle.v.angle,
            energy=status.energy,
            smell=smell,
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
                "eaten_preys": jnp.ravel(tactile_info.eaten_preys),
            },
        )
        predator_eat_timer = jnp.where(
            predator_energy_gain > 0,
            self._predator_eat_interval,
            state.predator_eat_timer - 1,
        )
        state = CFPredatorState(
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
            n_born_predators=state.n_born_predators,
            predator_eat_timer=predator_eat_timer,
        )
        return state, timestep

    def reset(  # type: ignore
        self,
        key: chex.PRNGKey,
    ) -> tuple[CFPredatorState, TimeStep[CFObsWithSmell]]:
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
        is_active = physics.circle.is_active
        n_preys = jnp.sum(is_active[: self._n_max_preys])
        n_predators = jnp.sum(is_active[self._n_max_preys :])
        unique_id = _init_uniqueid(
            n_preys,
            self._n_max_preys,
            n_predators,
            self._n_max_predators,
        )
        state = CFPredatorState(
            physics=physics,
            solver=self._physics.init_solver(),
            agent_loc=agent_loc,
            food_loc=food_loc,
            food_num=food_num,
            key=key,
            step=jnp.array(0, dtype=jnp.int32),
            unique_id=unique_id,
            status=status,
            n_born_agents=n_preys,
            n_born_predators=n_predators,
            predator_eat_timer=jnp.zeros(self._n_max_predators, dtype=jnp.int32),
        )
        sensor_obs = self._sensor_obs(stated=physics)  # type: ignore
        N = self.n_max_agents
        # Smell
        obs = CFObsWithSmell(
            sensor=sensor_obs.reshape(-1, self._n_sensors, self._n_obj),
            collision=jnp.zeros((N, self._n_obj, self._n_tactile_bins), dtype=bool),
            angle=physics.circle.p.angle,
            velocity=physics.circle.v.xy,
            angular_velocity=physics.circle.v.angle,
            energy=state.status.energy,
            smell=self._smell(physics.circle),
        )
        # They shouldn't encount now
        timestep = TimeStep(encount=jnp.zeros((N, N), dtype=bool), obs=obs)
        return state, timestep
