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
    thin_polygon_raycast,
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
    _make_physics_impl,
    _nonzero,
    _SensorFn,
    get_tactile,
    nstep,
)
from emevo.environments.env_utils import (
    CircleCoordinate,
    FoodNumState,
    LocatingState,
    SquareCoordinate,
    loc_gaussian,
    place,
)

Self = Any


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
    rc = thin_polygon_raycast(1.0, p1, p2, shaped.triangle, stated.triangle)
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
        obstacle_method: Literal["random", "fixed"] = "random",
        n_obstacles: int = 4,
        obstacle_size: float = 20.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs, _n_additional_objs=1)
        self._obstacle_damage = obstacle_damage
        self._n_obstacles = n_obstacles
        self._obstacle_method = obstacle_method
        self._obstacle_size = obstacle_size

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
        # TODO: Add obstacles
        a = Vec2d(0.0, self._obstacle_size)
        b = a.rotated_degrees(240) + a
        c = a.rotate_degrees(120) + b
        if self._obstacle_method == "random":
            for i in range(self._n_obstacles):
                builder.add_polygon()
        else:
            raise ValueError("Unsupported")
        return builder.build()
