from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from phyjax2d import Circle, Color, Raycast, ShapeDict
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
    AGENT_COLOR,
    FOOD_COLOR,
    MAX_ANGULAR_VELOCITY,
    MAX_VELOCITY,
    CFObs,
    CFState,
    CircleForaging,
    _get_sensors,
    _SensorFn,
    get_tactile,
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
PREDATOR_COLOR: Color = Color(6, 214, 160)


def _observe_closest(
    shaped: ShapeDict,
    circle_agent: Circle,
    circle_predator: Circle,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
    state_agent: State,
    state_predator: State,
) -> jax.Array:
    def cr(shape: Circle, state: State) -> Raycast:
        return circle_raycast(0.0, 1.0, p1, p2, shape, state)

    rc = cr(shaped.circle, stated.circle)
    to_c = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = cr(shaped.static_circle, stated.static_circle)
    to_sc = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = segment_raycast(1.0, p1, p2, shaped.segment, stated.segment)
    to_seg = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    obs = jnp.concatenate(
        jax.tree_util.tree_map(
            lambda arr: jnp.max(arr, keepdims=True),
            (to_c, to_sc, to_seg),
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

    def step(
        self,
        state: CFState[Status],
        action: ArrayLike,
    ) -> tuple[CFState[Status], TimeStep[CFObs]]:
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
        c2c = self._physics.get_contact_mat("circle", "circle", contacts)
        c2sc = self._physics.get_contact_mat("circle", "static_circle", contacts)
        seg2c = self._physics.get_contact_mat("segment", "circle", contacts)
        # Get tactile obs
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
        prey_tactile, _ = get_tactile(
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
        predator_prey_tactile, _ = get_tactile(
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
        self_tactile = jnp.concatenate(())
        tactile = jnp.concatenate(
            (prey_tactile > 0, food_tactile > 0, wall_tactile > 0),
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
        energy_gain = jnp.sum(n_ate * self._food_energy_coef, axis=1)
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
        # Construct obs
        obs = CFObs(
            sensor=sensor_obs.reshape(-1, self._n_sensors, self._n_obj),
            collision=tactile,
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
                "n_ate_food": n_ate,  # (N_AGENT, N_LABEL)
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
