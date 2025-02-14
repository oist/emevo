from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.env import Status, TimeStep, Visualizer
from emevo.environments.circle_foraging import (
    NOWHERE,
    CFObs,
    CFState,
    CircleForaging,
    _first_n_true,
    _get_sensors,
    _make_physics_impl,
    _SensorFn,
    get_tactile,
    nstep,
)
from emevo.environments.env_utils import FoodNumState, LocatingState
from emevo.phyjax2d import Color, ShapeDict
from emevo.phyjax2d import Space as Physics
from emevo.phyjax2d import (
    StateDict,
    Vec2d,
    circle_raycast,
    segment_raycast,
    thin_polygon_raycast,
)

Self = Any
OBSTACLE_COLOR: Color = Color(2, 204, 254)


def _observe_closest(
    shaped: ShapeDict,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
) -> jax.Array:
    rc = circle_raycast(0.0, 1.0, p1, p2, shaped.circle, stated.circle)
    to_c = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = circle_raycast(0.0, 1.0, p1, p2, shaped.static_circle, stated.static_circle)
    to_sc = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = segment_raycast(1.0, p1, p2, shaped.segment, stated.segment)
    to_seg = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = thin_polygon_raycast(
        1.0,
        p1,
        p2,
        shaped.static_triangle,
        stated.static_triangle,
    )
    to_tri = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    obs = jnp.concatenate(
        jax.tree_util.tree_map(
            lambda arr: jnp.max(arr, keepdims=True),
            (to_c, to_sc, to_seg, to_tri),
        ),
    )
    return jnp.where(obs == jnp.max(obs, axis=-1, keepdims=True), obs, -1.0)


_vmap_obs_closest = jax.vmap(_observe_closest, in_axes=(None, 0, 0, None))


def get_sensor_obs(
    shaped: ShapeDict,
    n_sensors: int,
    sensor_range: tuple[float, float],
    sensor_length: float,
    n_food_labels: int | None,
    stated: StateDict,
) -> jax.Array:
    assert stated.circle is not None
    p1, p2 = _get_sensors(
        shaped.circle,
        n_sensors,
        sensor_range,
        sensor_length,
        stated.circle,
    )
    if n_food_labels is None:
        return _vmap_obs_closest(shaped, p1, p2, stated)
    else:
        # TODO: Food label
        return _vmap_obs_closest(shaped, p1, p2, stated)


class CircleForagingWithObstacle(CircleForaging):
    def __init__(
        self,
        obstacle_damage: float = 10.0,
        n_obstacles: int = 4,
        obstacle_size: float = 20.0,
        **kwargs,
    ) -> None:
        self._obstacle_damage = obstacle_damage
        self._n_obstacles = n_obstacles
        self._obstacle_size = obstacle_size
        super().__init__(**kwargs, _n_additional_objs=1)
        xlim, ylim = self._coordinate.bbox()
        xlen, ylen = xlim[1] - xlim[0], ylim[1] - ylim[0]
        n_hol_blocks = xlen // obstacle_size
        n_vert_blocks = ylen // obstacle_size
        n_max_obstacles = n_hol_blocks * n_vert_blocks
        assert n_obstacles <= n_max_obstacles, (
            f"Too many obstacles! Max: {n_max_obstacles}, Current: {n_obstacles}"
        )
        self._n_hol_blocks = n_hol_blocks
        self._n_vert_blocks = n_vert_blocks
        self._hol_block_size = xlen / n_hol_blocks
        self._vert_block_size = ylen / n_vert_blocks

    def step(
        self,
        state: CFState[Status],
        action: ArrayLike,
    ) -> tuple[CFState[Status], TimeStep[CFObs]]:
        # Add force
        act = jax.vmap(self.act_space.clip)(jnp.array(action))
        f1_raw = jax.lax.slice_in_dim(act, 0, 1, axis=-1)
        f2_raw = jax.lax.slice_in_dim(act, 1, 2, axis=-1)
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
        tri2c = self._physics.get_contact_mat("static_triangle", "circle", contacts)
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
        obs_tactile, _ = get_tactile(
            self._n_tactile_bins,
            stated.circle,
            stated.static_triangle,
            tri2c.transpose(),
        )
        collision = jnp.concatenate(
            (ag_tactile > 0, food_tactile > 0, wall_tactile > 0, obs_tactile > 0),
            axis=1,
        )
        # Gather sensor obs
        sensor_obs = self._sensor_obs(stated=stated)  # type: ignore
        # energy_delta = food - coef * |force| - obstacle_damage * obs_tactile
        force_norm = jnp.sqrt(f1_raw**2 + f2_raw**2).ravel()
        energy_consumption = (
            self._force_energy_consumption * force_norm + self._basic_energy_consumption
        )
        damage = (
            jnp.max(obs_tactile.reshape(self.n_max_agents, -1), axis=-1)
            * self._obstacle_damage
        )
        n_ate = jnp.sum(food_tactile[:, :, self._foraging_indices], axis=-1)
        energy_gain = jnp.sum(n_ate * self._food_energy_coef, axis=1)
        energy_delta = energy_gain - energy_consumption - damage
        # Remove and regenerate foods
        key, food_key = jax.random.split(state.key)
        eaten = jnp.max(ft_raw[:, :, :, self._foraging_indices], axis=(0, 3)) > 0
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
                "obstacle_damage": damage,
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

    def _make_sensor_fn(self, observe_food_label: bool) -> _SensorFn:
        if observe_food_label:
            raise AssertionError("unsupported")
        else:
            return jax.jit(
                functools.partial(
                    get_sensor_obs,
                    shaped=self._physics.shaped,
                    n_sensors=self._n_sensors,
                    sensor_range=self._sensor_range_tuple,
                    sensor_length=self._sensor_length,
                    n_food_labels=None,
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
        builder = _make_physics_impl(
            dt=dt,
            coordinate=self._coordinate,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
            n_velocity_iter=n_velocity_iter,
            n_position_iter=n_position_iter,
            n_max_agents=self.n_max_agents,
            n_max_foods=self._n_max_foods,
            agent_radius=self._agent_radius,
            food_radius=self._food_radius,
            obstacles=obstacles,
        )
        # Radius of the inner circle
        r = (3**0.5) / 6 * self._obstacle_size
        a = Vec2d(self._obstacle_size * 0.5, -r)
        b = Vec2d(0.0, r * 2.0)
        c = Vec2d(-self._obstacle_size * 0.5, -r)
        triangle = [a, b, c]
        for _ in range(self._n_obstacles):
            builder.add_polygon(
                points=triangle,
                friction=0.6,
                elasticity=0.6,
                is_static=True,
                color=OBSTACLE_COLOR,
            )
        return builder.build()

    def _initialize_physics_state(
        self,
        key: chex.PRNGKey,
    ) -> tuple[StateDict, LocatingState, list[LocatingState], list[FoodNumState]]:
        stated = self._physics.shaped.zeros_state()
        assert stated.circle is not None
        # Move all circle to the invisiable area
        stated = stated.nested_replace(
            "circle.p.xy",
            jnp.ones_like(stated.circle.p.xy) * NOWHERE,
        )
        stated = stated.nested_replace(
            "static_circle.p.xy",
            jnp.ones_like(stated.static_circle.p.xy) * NOWHERE,
        )

        # Place obstacles
        key, obstacle_place_key, obstacle_angle_key = jax.random.split(key, 3)
        block_indices = jax.random.choice(
            obstacle_place_key,
            self._n_hol_blocks * self._n_vert_blocks,
            shape=(self._n_obstacles,),
            replace=False,
        )
        obs_x_indices = block_indices % self._n_hol_blocks
        obs_y_indices = block_indices // self._n_hol_blocks
        obs_x = obs_x_indices * self._hol_block_size + self._hol_block_size * 0.5
        obs_y = obs_y_indices * self._vert_block_size + self._vert_block_size * 0.5
        stated = stated.nested_replace(
            "static_triangle.p.xy",
            jnp.stack((obs_x, obs_y), axis=1),
        )
        stated = stated.nested_replace(
            "static_triangle.p.angle",
            jax.random.uniform(obstacle_angle_key, shape=(self._n_obstacles,)),
        )
        # Place agents
        key, *agent_keys = jax.random.split(key, self._n_initial_agents + 1)
        n_agents = 0
        agentloc_state = self._initial_agentloc_state
        is_active = []
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
                n_agents += 1
            is_active.append(ok)

        if n_agents < self._n_initial_agents:
            diff = self._n_initial_agents - n_agents
            warnings.warn(f"Failed to place {diff} agents!", stacklevel=1)

        # Set is_active
        is_active_c = jnp.concatenate(
            (
                jnp.array(is_active),
                jnp.zeros(self.n_max_agents - n_agents, dtype=bool),
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
        for i, food_key_i in enumerate(jax.random.split(key, self._n_food_sources)):
            n_initial = self._food_num_fns[i].initial
            xy, ok = self._place_food_fns[i](
                loc_state=foodloc_states[i],
                n_max_placement=n_initial,
                key=food_key_i,
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

    def visualizer(
        self,
        state: CFState[Status],
        figsize: tuple[float, float] | None = None,
        sensor_index: int | None = None,
        no_sensor: bool = False,
        backend: str = "pyglet",
        **kwargs,
    ) -> Visualizer[StateDict]:
        """Create a visualizer for the environment"""
        from emevo.phyjax2d import moderngl_vis

        if sensor_index is not None:
            self._sensor_index = sensor_index

        if sensor_index is None:
            sensor_fn = self._get_sensors_for_vis
        else:

            def sensor_fn(stated: StateDict) -> tuple[jax.Array, jax.Array]:
                return self._get_selected_sensor(stated, self._sensor_index)

        return moderngl_vis.MglVisualizer(
            x_range=self._x_range,
            y_range=self._y_range,
            space=self._physics,
            stated=state.physics,
            sc_color=self._food_color,
            figsize=figsize,
            backend=backend,
            sensor_fn=None if no_sensor else sensor_fn,  # type: ignore
            **kwargs,
        )
