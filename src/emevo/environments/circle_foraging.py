from __future__ import annotations

import enum
import functools
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from emevo.env import (
    Env,
    Status,
    TimeStep,
    UniqueID,
    Visualizer,
    init_status,
    init_uniqueid,
)
from emevo.environments.env_utils import (
    CircleCoordinate,
    FoodNum,
    FoodNumFn,
    FoodNumState,
    Locating,
    LocatingFn,
    LocatingState,
    LocGaussian,
    SquareCoordinate,
    check_points_are_far_from_other_foods,
    first_to_nth_true,
    place,
    place_multi,
)
from emevo.phyjax2d import (
    Circle,
    Color,
    Position,
    ShapeDict,
)
from emevo.phyjax2d import Space as Physics
from emevo.phyjax2d import (
    SpaceBuilder,
    State,
    StateDict,
    Vec2d,
    Velocity,
    VelocitySolver,
    circle_raycast,
    get_relative_angle,
    make_approx_circle,
    make_square_segments,
    segment_raycast,
)
from emevo.phyjax2d import step as physics_step
from emevo.spaces import BoxSpace, NamedTupleSpace

MAX_ANGULAR_VELOCITY: float = float(np.pi)
MAX_VELOCITY: float = 10.0
AGENT_COLOR: Color = Color(11, 95, 174)
FOOD_COLOR: Color = Color(27, 121, 35)
HEAD_COLOR: Color = Color(167, 37, 193)
NOWHERE: float = 0.0
N_OBJECTS: int = 3


class CFObs(NamedTuple):
    """Observation of an agent."""

    sensor: jax.Array
    collision: jax.Array
    velocity: jax.Array
    angle: jax.Array
    angular_velocity: jax.Array
    energy: jax.Array

    def as_array(self) -> jax.Array:
        return jnp.concatenate(
            (
                self.sensor.reshape(self.sensor.shape[0], -1),
                self.collision.reshape(self.collision.shape[0], -1).astype(jnp.float32),
                self.velocity,
                jnp.expand_dims(self.angle, axis=1),
                jnp.expand_dims(self.angular_velocity, axis=1),
                jnp.expand_dims(self.energy, axis=1),
            ),
            axis=1,
        )


S = TypeVar("S", bound=Status)


@chex.dataclass
class CFState(Generic[S]):
    physics: StateDict
    solver: VelocitySolver
    food_num: list[FoodNumState]
    agent_loc: LocatingState
    food_loc: list[LocatingState]
    key: chex.PRNGKey
    step: jax.Array
    unique_id: UniqueID
    status: S
    n_born_agents: jax.Array

    @property
    def stated(self) -> StateDict:
        return self.physics

    def is_extinct(self) -> bool:
        return jnp.logical_not(jnp.any(self.unique_id.is_active())).item()


class Obstacle(str, enum.Enum):
    NONE = "none"
    CENTER = "center"
    ONE_FOURTH = "one-fourth"
    ONE_THIRD = "one-third"
    CENTER_HALF = "center-half"
    CENTER_TWO_THIRDS = "center-two-thirds"

    def as_list(
        self,
        width: float,
        height: float,
    ) -> list[tuple[Vec2d, Vec2d]]:
        # xmin, xmax, ymin, ymax
        if self == Obstacle.NONE:
            return []
        # Vertical line that divides the room into two
        elif self == Obstacle.CENTER:
            return [(Vec2d(width / 2, 0.0), Vec2d(width / 2, height))]
        elif self == Obstacle.ONE_FOURTH:
            return [(Vec2d(width / 4, 0.0), Vec2d(width / 4, height))]
        elif self == Obstacle.ONE_THIRD:
            return [(Vec2d(width / 3, 0.0), Vec2d(width / 3, height))]
        elif self == Obstacle.CENTER_HALF:
            return [(Vec2d(width / 2, height / 2), Vec2d(width / 2, height))]
        elif self == Obstacle.CENTER_TWO_THIRDS:
            return [(Vec2d(width / 2, height / 3), Vec2d(width / 2, height))]
        else:
            raise ValueError(f"Unsupported Obstacle: {self}")


class SensorRange(str, enum.Enum):
    NARROW = "narrow"
    WIDE = "wide"  # 120 deg
    WIDE_160D = "wide-160d"
    WIDE_180D = "wide-180d"
    ALL = "all"

    def as_tuple(self) -> tuple[float, float]:
        if self == SensorRange.NARROW:
            return -30.0, 30.0
        elif self == SensorRange.WIDE:
            return -60.0, 60.0
        elif self == SensorRange.WIDE_160D:
            return -80.0, 80.0
        elif self == SensorRange.WIDE_180D:
            return -90.0, 90.0
        else:
            return -180.0, 180.0


def _get_num_or_loc_fn(
    arg: str | tuple | list | Callable[..., Any],
    enum_type: Callable[..., Callable[..., Any]],
    default_args: dict[str, tuple[Any, ...]],
    placement_args: dict[str, tuple[Any, ...]] | None = None,
) -> Any:
    if callable(arg):
        return arg
    if isinstance(arg, str):
        return enum_type(arg)(*default_args[arg])
    elif isinstance(arg, tuple) or isinstance(arg, list):
        name, *args = arg
        if placement_args is None or name not in placement_args:
            first_args = ()
        else:
            first_args = placement_args[name]
        return enum_type(name)(*first_args, *args)
    else:
        raise ValueError(f"Invalid value in _get_num_or_loc_fn {arg}")


def _make_physics_impl(
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
) -> SpaceBuilder:
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
        walls = make_square_segments(
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
    return builder


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
    obs = jnp.concatenate(
        jax.tree_util.tree_map(
            lambda arr: jnp.max(arr, keepdims=True),
            (to_c, to_sc, to_seg),
        ),
    )
    return jnp.where(obs == jnp.max(obs, axis=-1, keepdims=True), obs, -1.0)


def _observe_closest_with_food_labels(
    n_food_labels: int,
    shaped: ShapeDict,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
) -> jax.Array:
    rc = circle_raycast(0.0, 1.0, p1, p2, shaped.circle, stated.circle)
    to_c = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    rc = circle_raycast(0.0, 1.0, p1, p2, shaped.static_circle, stated.static_circle)
    to_sc = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    foodlabel_onehot = jax.nn.one_hot(
        stated.static_circle.label,
        n_food_labels,
        dtype=bool,
    )
    to_sc_all = jnp.where(foodlabel_onehot, jnp.expand_dims(to_sc, axis=1), -1.0)
    rc = segment_raycast(1.0, p1, p2, shaped.segment, stated.segment)
    to_seg = jnp.where(rc.hit, 1.0 - rc.fraction, -1.0)
    obs = jnp.concatenate(
        (
            jnp.max(to_c, keepdims=True),
            # (N_FOOD, N_LABEL) -> (N_LABEL,)
            jnp.max(to_sc_all, axis=0),
            jnp.max(to_seg, keepdims=True),
        )
    )
    return jnp.where(obs == jnp.max(obs, axis=-1, keepdims=True), obs, -1.0)


_vmap_obs_closest = jax.vmap(_observe_closest, in_axes=(None, 0, 0, None))
_vmap_obs_closest_with_food = jax.vmap(
    _observe_closest_with_food_labels,
    in_axes=(None, None, 0, 0, None),
)


def _first_n_true(boolean_array: jax.Array, n: jax.Array) -> jax.Array:
    return jnp.logical_and(boolean_array, jnp.cumsum(boolean_array) <= n)


def _nonzero(arr: jax.Array, n: int) -> jax.Array:
    """Similar to jax.numpy.nonzero, but simpler"""
    cums = jnp.cumsum(arr)
    bincount = jnp.zeros(n, dtype=jnp.int32).at[cums].add(1)
    return jnp.cumsum(bincount)


def _get_sensors(
    shape: Circle,
    n_sensors: int,
    sensor_range: tuple[float, float],
    sensor_length: float,
    state: State,
) -> tuple[jax.Array, jax.Array]:
    radius = shape.radius
    p1 = jnp.stack((jnp.zeros_like(radius), radius), axis=1)  # (N, 2)
    p1 = jnp.repeat(p1, n_sensors, axis=0)  # (N x M, 2)
    p2 = p1 + jnp.array([0.0, sensor_length])  # (N x M, 2)
    sensor_rad = jnp.deg2rad(jnp.linspace(*sensor_range, n_sensors))
    sensor_p = Position(
        angle=jax.vmap(lambda x: x + sensor_rad)(state.p.angle).ravel(),
        xy=jnp.repeat(state.p.xy, n_sensors, axis=0),
    )
    p1 = sensor_p.transform(p1)
    p2 = sensor_p.transform(p2)
    return p1, p2


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
        return _vmap_obs_closest_with_food(n_food_labels, shaped, p1, p2, stated)


@functools.partial(jax.vmap, in_axes=(0, None))
def _search_bin(value: jax.Array, bins: jax.Array) -> jax.Array:
    smaller = value <= bins[1:]
    larger = bins[:-1] <= value
    return jnp.logical_and(smaller, larger)


def get_tactile(
    n_bins: int,
    s1: State,
    s2: State,
    collision_mat: jax.Array,
    shift: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    n, m = collision_mat.shape
    rel_angle = get_relative_angle(s1, s2)  # [0, 2π]  (N, M)
    weights = (jnp.pi * 2 / n_bins) * jnp.arange(n_bins + 1)  # [0, ..., 2π]
    # If shift > 0, shift angles
    angle_shifted = (rel_angle.ravel() + shift) % (jnp.pi * 2)
    in_range = _search_bin(angle_shifted, weights).reshape(n, m, n_bins)
    tactile_raw = in_range * jnp.expand_dims(collision_mat, axis=2)  # (N, M, B)
    tactile = jnp.sum(tactile_raw, axis=1, keepdims=True)  # (N, 1, B)
    return tactile, jnp.expand_dims(tactile_raw, axis=2)  # (N, M, 1, B)


def _food_tactile_with_labels(
    n_bins: int,
    n_food_sources: int,
    food_labels: jax.Array,
    s1: State,
    s2: State,
    collision_mat: jax.Array,
    shift: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    n, m = collision_mat.shape
    rel_angle = get_relative_angle(s1, s2)  # [0, 2π]
    weights = (jnp.pi * 2 / n_bins) * jnp.arange(n_bins + 1)  # [0, ..., 2π]
    # If shift > 0, shift angles
    angle_shifted = (rel_angle.ravel() + shift) % (jnp.pi * 2)
    in_range = _search_bin(angle_shifted, weights).reshape(n, m, n_bins)
    in_range_masked = in_range * jnp.expand_dims(collision_mat, axis=2)
    onehot = jax.nn.one_hot(food_labels, n_food_sources, dtype=bool)
    expanded_onehot = onehot.reshape(1, *onehot.shape, 1)  # (1, M, L, 1)
    expanded_in_range = jnp.expand_dims(in_range_masked, axis=2)  # (N, M, 1, B)
    tactile_raw = expanded_in_range * expanded_onehot  # (N, M, L, B)
    tactile = jnp.sum(tactile_raw, axis=1)  # (N, L, B)
    return tactile, tactile_raw


@functools.partial(jax.jit, static_argnums=(0, 1))
def nstep(
    n: int,
    space: Physics,
    stated: StateDict,
    solver: VelocitySolver,
) -> tuple[StateDict, VelocitySolver, jax.Array]:
    def body(
        stated_and_solver: tuple[StateDict, VelocitySolver],
        _: jax.Array,
    ) -> tuple[tuple[StateDict, VelocitySolver], jax.Array]:
        state, solver, contact = physics_step(space, *stated_and_solver)
        return (state, solver), contact.penetration >= 0.0

    (state, solver), contacts = jax.lax.scan(body, (stated, solver), jnp.zeros(n))
    return state, solver, contacts


def _set_b2a(
    xy_a: jax.Array,
    flag_a: jax.Array,
    xy_b: jax.Array,
    flag_b: jax.Array,
) -> jax.Array:
    """Do `xy_a[flag_a] = xy_b[flag_b]`, but compatible with jax.jit"""
    a_len = xy_a.shape[0]
    a_idx = _nonzero(flag_a, a_len + 1)
    b_idx = _nonzero(flag_b, a_len + 1)
    xy_b_with_sentinel = jnp.concatenate((xy_b, jnp.zeros((1, xy_b.shape[1]))))
    # Fill xy_a[flag_a] with 0
    xy_a_reset = jnp.where(jnp.expand_dims(flag_a, axis=1), 0.0, xy_a)
    return xy_a_reset.at[a_idx].add(xy_b_with_sentinel[b_idx])


def _make_food_energy_coef_array(
    food_energy_coef: Iterable[float | tuple[float, ...]],
) -> jax.Array:
    has_tuple = any([isinstance(fec, tuple) for fec in food_energy_coef])
    if has_tuple:
        length = [len(fec) if isinstance(fec, tuple) else 1 for fec in food_energy_coef]
        lcm = math.lcm(*length)
        elements = []
        for fec in food_energy_coef:
            if isinstance(fec, tuple):
                elements.append(fec * (lcm // len(fec)))
            else:
                elements.append([fec] * lcm)
        return jnp.array(elements)
    else:
        return jnp.expand_dims(
            jnp.array(list(food_energy_coef)),
            axis=0,
        )


_MaybeLocatingFn = LocatingFn | str | tuple[str, ...]
_MaybeNumFn = FoodNumFn | str | tuple[str, ...]
_SensorFn = Callable[[StateDict], jax.Array]

_MOUTH_RANGE = Literal[
    "full",
    "front",
    "front-wide",
    "right",
]


class CircleForaging(Env):
    def __init__(
        self,
        *,
        n_initial_agents: int = 6,
        n_max_agents: int = 100,
        n_max_foods: int = 40,
        n_food_sources: int = 1,
        food_num_fn: _MaybeNumFn | list[_MaybeNumFn] = "constant",
        food_loc_fn: _MaybeLocatingFn | list[_MaybeLocatingFn] = "gaussian",
        agent_loc_fn: LocatingFn | str | tuple[str, ...] = "uniform",
        food_energy_coef: Iterable[float | tuple[float, ...]] = (1.0,),
        food_color: Iterable[tuple] = (FOOD_COLOR,),
        xlim: tuple[float, float] = (0.0, 200.0),
        ylim: tuple[float, float] = (0.0, 200.0),
        env_radius: float = 120.0,
        env_shape: Literal["square", "circle"] = "square",
        obstacles: list[tuple[Vec2d, Vec2d]] | str = "none",
        newborn_loc: Literal["neighbor", "uniform"] = "neighbor",
        mouth_range: _MOUTH_RANGE | list[int] = "front",
        neighbor_stddev: float = 40.0,
        n_agent_sensors: int = 16,
        n_tactile_bins: int = 6,
        tactile_shift: float = 0.0,
        sensor_length: float = 100.0,
        sensor_range: tuple[float, float] | SensorRange = SensorRange.WIDE,
        agent_radius: float = 10.0,
        food_radius: float = 4.0,
        foodloc_interval: int = 1000,
        fec_intervals: tuple[int, ...] = (1,),
        dt: float = 0.1,
        linear_damping: float = 0.8,
        angular_damping: float = 0.6,
        max_force: float = 40.0,
        min_force: float = -20.0,
        init_energy: float = 20.0,
        energy_capacity: float = 100.0,
        observe_food_label: bool = False,
        force_energy_consumption: float = 0.01 / 40.0,
        basic_energy_consumption: float = 0.0,
        energy_share_ratio: float = 0.4,
        foods_min_dist: float = 0.0,
        n_velocity_iter: int = 6,
        n_position_iter: int = 2,
        n_physics_iter: int = 5,
        max_place_attempts: int = 10,
        n_max_food_regen: int = 20,
        random_angle: bool = True,  # False when debugging/testing
        _n_additional_objs: int = 0,  # Used by child classes (e.g., predator)
    ) -> None:
        # Coordinate and range
        if env_shape == "square":
            self._coordinate = SquareCoordinate(xlim, ylim)
        elif env_shape == "circle":
            self._coordinate = CircleCoordinate((env_radius, env_radius), env_radius)
        else:
            raise ValueError(f"Unsupported env_shape {env_shape}")

        self._xlim, self._ylim = self._coordinate.bbox()
        self._x_range = self._xlim[1] - self._xlim[0]
        self._y_range = self._ylim[1] - self._ylim[0]
        # Food and body placing functions
        self._agent_radius = agent_radius
        self._food_radius = food_radius
        self._foodloc_interval = foodloc_interval
        self._n_food_sources = n_food_sources
        self._food_loc_fns, self._initial_foodloc_states = [], []
        self._food_energy_coef = _make_food_energy_coef_array(food_energy_coef)
        self._fec_intervals = jnp.array(fec_intervals, dtype=jnp.int32)
        self._food_num_fns, self._initial_foodnum_states = [], []
        self._foods_min_dist = foods_min_dist
        if n_food_sources > 1:
            assert isinstance(food_loc_fn, list | tuple)
            assert n_food_sources == len(food_loc_fn)
            assert isinstance(food_num_fn, list | tuple)
            assert n_food_sources == len(food_num_fn)
        else:
            food_loc_fn, food_num_fn = [food_loc_fn], [food_num_fn]  # type: ignore
        for maybe_loc_fn in food_loc_fn:  # type: ignore
            fn, state = self._make_food_loc_fn(maybe_loc_fn)
            self._food_loc_fns.append(fn)
            self._initial_foodloc_states.append(state)
        for maybe_num_fn in food_num_fn:  # type: ignore
            fn, state = self._make_food_num_fn(maybe_num_fn)
            self._food_num_fns.append(fn)
            self._initial_foodnum_states.append(state)
        self._agent_loc_fn, self._initial_agentloc_state = self._make_agent_loc_fn(
            agent_loc_fn
        )
        # Foraging
        if mouth_range == "full":
            self._foraging_indices = tuple(range(n_tactile_bins))
        elif mouth_range == "front":
            # Left and right \ /
            self._foraging_indices = 0, n_tactile_bins - 1
        elif mouth_range == "front-wide":
            # Leftx2 and rightx2 \\ //
            assert n_tactile_bins >= 4
            self._foraging_indices = 0, 1, n_tactile_bins - 2, n_tactile_bins - 1
        elif mouth_range == "right":
            self._foraging_indices = (n_tactile_bins - 1,)
        else:
            self._foraging_indices = tuple(mouth_range)
        # Energy
        self._force_energy_consumption = force_energy_consumption
        self._basic_energy_consumption = basic_energy_consumption
        self._init_energy = init_energy
        self._energy_capacity = energy_capacity
        self._energy_share_ratio = energy_share_ratio
        # Initial numbers
        assert n_max_agents > n_initial_agents
        self._n_initial_foods = sum([num_fn.initial for num_fn in self._food_num_fns])
        assert n_max_foods >= self._n_initial_foods
        self._n_initial_agents = n_initial_agents
        self.n_max_agents = n_max_agents
        self._n_max_foods = n_max_foods
        self._max_place_attempts = max_place_attempts
        self._n_max_food_regen = n_max_food_regen
        self._random_angle = random_angle
        # Physics
        if isinstance(obstacles, str):
            obs_list = Obstacle(obstacles).as_list(self._x_range, self._y_range)
        else:
            obs_list = obstacles

        self._physics = self._make_physics(
            dt=dt,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
            n_velocity_iter=n_velocity_iter,
            n_position_iter=n_position_iter,
            obstacles=obs_list,
        )
        self._agent_indices = jnp.arange(n_max_agents)
        self._food_indices = jnp.arange(n_max_foods)
        self._n_physics_iter = n_physics_iter
        # Obs
        self._n_sensors = n_agent_sensors
        self._n_tactile_bins = n_tactile_bins
        self._tactile_shift = (tactile_shift / 360.0) * 2.0 * jnp.pi
        del tactile_shift
        # Some cached constants
        act_p1 = Vec2d(0, agent_radius).rotated(np.pi * 0.75)
        act_p2 = Vec2d(0, agent_radius).rotated(-np.pi * 0.75)
        self._act_p1 = jnp.tile(jnp.array(act_p1), (self.n_max_agents, 1))
        self._act_p2 = jnp.tile(jnp.array(act_p2), (self.n_max_agents, 1))
        self._init_agent = jax.jit(
            functools.partial(
                place,
                n_trial=max_place_attempts,
                radius=self._agent_radius,
                coordinate=self._coordinate,
                loc_fn=self._agent_loc_fn,
                shaped=self._physics.shaped,
            )
        )

        self._place_food_fns = []
        for loc_fn in self._food_loc_fns:
            place_fn = jax.jit(
                functools.partial(
                    place_multi,
                    n_trial=n_max_food_regen,
                    radius=self._food_radius,
                    coordinate=self._coordinate,
                    loc_fn=loc_fn,
                    shaped=self._physics.shaped,
                )
            )
            self._place_food_fns.append(place_fn)

        if newborn_loc == "uniform":

            def place_newborn_uniform(
                state: LocatingState,
                stated: StateDict,
                key: chex.PRNGKey,
                _: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return place(
                    n_trial=self._max_place_attempts,
                    radius=self._agent_radius,
                    coordinate=self._coordinate,
                    loc_fn=self._agent_loc_fn,
                    shaped=self._physics.shaped,
                    loc_state=state,
                    key=key,
                    n_steps=0,
                    stated=stated,
                )

            self._place_newborn = jax.vmap(
                place_newborn_uniform,
                in_axes=(None, None, 0, None),
            )

        elif newborn_loc == "neighbor":

            def place_newborn_neighbor(
                state: LocatingState,
                stated: StateDict,
                key: chex.PRNGKey,
                agent_loc: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                loc_fn = LocGaussian(
                    agent_loc,
                    jnp.ones_like(agent_loc) * neighbor_stddev,
                )

                return place(
                    n_trial=self._max_place_attempts,
                    radius=self._agent_radius,
                    coordinate=self._coordinate,
                    loc_fn=loc_fn,
                    shaped=self._physics.shaped,
                    loc_state=state,
                    key=key,
                    n_steps=0,
                    stated=stated,
                )

            self._place_newborn = jax.vmap(
                place_newborn_neighbor,
                in_axes=(None, None, 0, 0),
            )
        else:
            raise ValueError(f"Invalid newborn_loc: {newborn_loc}")
        if isinstance(sensor_range, SensorRange | str):
            self._sensor_range_tuple = SensorRange(sensor_range).as_tuple()
        else:
            self._sensor_range_tuple = sensor_range
        self._sensor_length = sensor_length

        self._sensor_obs = self._make_sensor_fn(observe_food_label)

        if observe_food_label:
            assert (
                self._n_food_sources > 1
            ), "n_food_sources should be larager than 1 to include food label obs"

            self._food_tactile = lambda labels, s1, s2, cmat: _food_tactile_with_labels(
                self._n_tactile_bins,
                self._n_food_sources,
                labels,
                s1,
                s2,
                cmat,
                shift=self._tactile_shift,
            )
            self._n_obj = N_OBJECTS + self._n_food_sources - 1 + _n_additional_objs

        else:
            self._food_tactile = lambda _, s1, s2, cmat: get_tactile(
                self._n_tactile_bins,
                s1,
                s2,
                cmat,
                shift=self._tactile_shift,
            )
            self._n_obj = N_OBJECTS + _n_additional_objs

        # Spaces
        self.act_space = BoxSpace(low=min_force, high=max_force, shape=(2,))
        self.obs_space = NamedTupleSpace(
            CFObs,
            sensor=BoxSpace(low=0.0, high=1.0, shape=(n_agent_sensors, self._n_obj)),
            collision=BoxSpace(low=0.0, high=1.0, shape=(self._n_obj, n_tactile_bins)),
            velocity=BoxSpace(low=-MAX_VELOCITY, high=MAX_VELOCITY, shape=(2,)),
            angle=BoxSpace(low=-2 * np.pi, high=2 * np.pi, shape=()),
            angular_velocity=BoxSpace(low=-np.pi / 10, high=np.pi / 10, shape=()),
            energy=BoxSpace(low=0.0, high=energy_capacity, shape=()),
        )

        # For visualization
        self._food_color = np.array(list(food_color))

        @jax.jit
        def get_sensors_for_vis(stated: StateDict) -> tuple[jax.Array, jax.Array]:
            return _get_sensors(
                state=stated.circle,
                shape=self._physics.shaped.circle,
                n_sensors=n_agent_sensors,
                sensor_range=self._sensor_range_tuple,
                sensor_length=sensor_length,
            )

        self._get_sensors_for_vis = get_sensors_for_vis

        # Sensor index
        self._sensor_index = 0

    @staticmethod
    def _make_food_num_fn(
        food_num_fn: str | tuple | FoodNumFn,
    ) -> tuple[FoodNumFn, FoodNumState]:
        return _get_num_or_loc_fn(
            food_num_fn,
            FoodNum,  # type: ignore
            {"constant": (10,), "linear": (10, 0.01), "logistic": (8, 0.01, 12)},
        )

    def _make_food_loc_fn(
        self,
        food_loc_fn: str | tuple | LocatingFn,
    ) -> tuple[LocatingFn, LocatingState]:
        return _get_num_or_loc_fn(
            food_loc_fn,
            Locating,  # type: ignore
            {
                "gaussian": (
                    (self._xlim[1] * 0.75, self._ylim[1] * 0.75),
                    (self._x_range * 0.1, self._y_range * 0.1),
                ),
                "gaussian-mixture": (
                    [0.5, 0.5],
                    [
                        (self._xlim[1] * 0.75, self._ylim[1] * 0.75),
                        (self._xlim[1] * 0.25, self._ylim[1] * 0.75),
                    ],
                    [(self._x_range * 0.1, self._y_range * 0.1)] * 2,
                ),
                "switching": (
                    self._foodloc_interval,
                    (
                        "gaussian",
                        (self._xlim[1] * 0.75, self._ylim[1] * 0.75),
                        (self._x_range * 0.1, self._y_range * 0.1),
                    ),
                    (
                        "gaussian",
                        (self._xlim[1] * 0.25, self._ylim[1] * 0.75),
                        (self._x_range * 0.1, self._y_range * 0.1),
                    ),
                ),
                "uniform": (self._coordinate,),
            },
            {
                "uniform-linear": (self._coordinate,),
            },
        )

    def _make_agent_loc_fn(
        self,
        init_loc_fn: str | tuple | LocatingFn,
    ) -> tuple[LocatingFn, LocatingState]:
        return _get_num_or_loc_fn(
            init_loc_fn,
            Locating,  # type: ignore
            {
                "gaussian": (
                    (self._xlim[1] * 0.25, self._ylim[1] * 0.25),
                    (self._x_range * 0.3, self._y_range * 0.3),
                ),
                "uniform": (self._coordinate,),
            },
        )

    def _get_selected_sensor(
        self,
        stated: StateDict,
        index: int,
    ) -> tuple[jax.Array, jax.Array]:
        p1, p2 = self._get_sensors_for_vis(stated)
        from_ = index * self._n_sensors
        to = (index + 1) * self._n_sensors
        zeros = jnp.ones_like(p1)
        p1 = zeros.at[from_:to].add(p1[from_:to])
        p2 = zeros.at[from_:to].add(p2[from_:to])
        return p1, p2

    def _make_sensor_fn(self, observe_food_label: bool) -> _SensorFn:
        if observe_food_label:
            return jax.jit(
                functools.partial(
                    get_sensor_obs,
                    shaped=self._physics.shaped,
                    n_sensors=self._n_sensors,
                    sensor_range=self._sensor_range_tuple,
                    sensor_length=self._sensor_length,
                    n_food_labels=self._n_food_sources,
                )
            )
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
            shift=self._tactile_shift,
        )
        wall_tactile, _ = get_tactile(
            self._n_tactile_bins,
            stated.circle,
            stated.segment,
            seg2c.transpose(),
            shift=self._tactile_shift,
        )
        collision = jnp.concatenate(
            (ag_tactile > 0, food_tactile > 0, wall_tactile > 0),
            axis=1,
        )
        # Gather sensor obs
        sensor_obs = self._sensor_obs(stated=stated)  # type: ignore
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

    def activate(
        self,
        state: CFState[Status],
        is_parent: jax.Array,
    ) -> tuple[CFState, jax.Array]:
        N = self.n_max_agents
        circle = state.physics.circle
        keys = jax.random.split(state.key, N + 2)
        new_xy, ok = self._place_newborn(
            state.agent_loc,
            state.physics,
            keys[2:],
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
        parent_indices = _nonzero(is_parent, N)
        # empty_indices := nonzero_indices(not(is_active)) + (N, N, N, ....)
        replaced_indices = _nonzero(is_replaced, N)
        # To use .at[].add, append (0, 0) to sampled xy
        new_xy_with_sentinel = jnp.concatenate((new_xy, jnp.zeros((1, 2))))
        xy = circle.p.xy.at[replaced_indices].add(new_xy_with_sentinel[parent_indices])
        if self._random_angle:
            new_angle = jax.random.uniform(keys[1]) * jnp.pi * 2.0
            angle = jnp.where(is_replaced, new_angle, circle.p.angle)
        else:
            angle = jnp.where(is_replaced, 0.0, circle.p.angle)
        p = Position(angle=angle, xy=xy)
        is_active = jnp.logical_or(is_replaced, circle.is_active)
        physics = replace(
            state.physics,
            circle=replace(circle, p=p, is_active=is_active),
        )
        unique_id = state.unique_id.activate(is_replaced)
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
            key=keys[0],
        )
        empty_id = jnp.ones_like(state.unique_id.unique_id) * -1
        unique_id_with_sentinel = jnp.concatenate(
            (state.unique_id.unique_id, jnp.zeros(1, dtype=jnp.int32))
        )
        parent_id = empty_id.at[replaced_indices].set(
            unique_id_with_sentinel[parent_indices]
        )
        return new_state, parent_id

    def deactivate(self, state: CFState, flag: jax.Array) -> CFState[Status]:
        expanded_flag = jnp.expand_dims(flag, axis=1)
        p_xy = jnp.where(expanded_flag, NOWHERE, state.physics.circle.p.xy)
        p = replace(state.physics.circle.p, xy=p_xy)
        v_xy = jnp.where(expanded_flag, 0.0, state.physics.circle.v.xy)
        v_angle = jnp.where(flag, 0.0, state.physics.circle.v.angle)
        v = Velocity(angle=v_angle, xy=v_xy)
        is_active = jnp.where(flag, False, state.physics.circle.is_active)
        circle = replace(state.physics.circle, p=p, v=v, is_active=is_active)
        physics = replace(state.physics, circle=circle)
        unique_id = state.unique_id.deactivate(flag)
        status = state.status.deactivate(flag)
        return replace(state, physics=physics, unique_id=unique_id, status=status)

    def reset(self, key: chex.PRNGKey) -> tuple[CFState[Status], TimeStep[CFObs]]:
        physics, agent_loc, food_loc, food_num = self._initialize_physics_state(key)
        N = self.n_max_agents
        n_agents = jnp.sum(physics.circle.is_active)
        unique_id = init_uniqueid(int(n_agents), N)
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
            n_born_agents=n_agents,
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
            # if foods_min_dist is given, compute distances to 'other' foods and
            # reject the posision if it's too close
            if self._foods_min_dist > 0.0:
                ok = ok & check_points_are_far_from_other_foods(
                    self._foods_min_dist,
                    i,
                    xy,
                    stated,
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

    def _remove_and_regenerate_foods(
        self,
        old_key: chex.PRNGKey,
        eaten_per_source: jax.Array,
        sd: StateDict,
        n_steps: jax.Array,
        food_num_states: list[FoodNumState],
        food_loc_states: list[LocatingState],
    ) -> tuple[StateDict, list[FoodNumState], list[LocatingState], jax.Array]:
        eaten = jnp.sum(eaten_per_source, axis=1) > 0
        # Remove foods
        xy = jnp.where(
            jnp.expand_dims(eaten, axis=1),
            jnp.ones_like(sd.static_circle.p.xy) * NOWHERE,
            sd.static_circle.p.xy,
        )
        is_active = jnp.logical_and(sd.static_circle.is_active, jnp.logical_not(eaten))
        n_eaten_per_source = jnp.sum(eaten_per_source, axis=0)
        sc = sd.static_circle
        # Regenerate food for each source
        n_generated_foods = jnp.zeros(self._n_food_sources, dtype=jnp.int32)
        keys = jax.random.split(old_key, self._n_food_sources)
        for i, key in enumerate(keys):
            food_num_states[i] = food_num_states[i].eaten(n_eaten_per_source[i])
            food_num = self._food_num_fns[i](n_steps, food_num_states[i])
            food_loc = food_loc_states[i]
            # (N_MAX_REGEN, 2), (N_MAX_REGEN,)
            new_food_xy, ok = self._place_food_fns[i](
                loc_state=food_loc,
                n_max_placement=food_num.n_max_recover(),
                key=key,
                n_steps=n_steps,
                stated=sd,
            )
            if self._foods_min_dist > 0.0:
                ok = ok & check_points_are_far_from_other_foods(
                    self._foods_min_dist,
                    i,
                    new_food_xy,
                    sd,
                )
            place = first_to_nth_true(jnp.logical_not(is_active), jnp.sum(ok))
            xy = _set_b2a(xy, place, new_food_xy, ok)
            is_active = jnp.logical_or(is_active, place)
            p = replace(sc.p, xy=xy)
            label = jnp.where(place, i, sc.label)
            sc = replace(sc, p=p, is_active=is_active, label=label)
            incr = jnp.sum(place)
            food_num_states[i] = food_num.recover(incr)
            food_loc_states[i] = food_loc.increment(incr)
            n_generated_foods = n_generated_foods.at[i].add(incr)
        return (
            replace(sd, static_circle=sc),
            food_num_states,
            food_loc_states,
            n_generated_foods,
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
        return builder.build()

    def visualizer(
        self,
        state: CFState[Status],
        figsize: tuple[float, float] | None = None,
        sensor_index: int | None = None,
        no_sensor: bool = False,
        backend: str = "pyglet",
        partial_range_x: float | None = None,
        partial_range_y: float | None = None,
        **kwargs,
    ) -> Visualizer[StateDict]:
        """Create a visualizer for the environment"""
        from emevo.environments import moderngl_vis

        if sensor_index is not None:
            self._sensor_index = sensor_index

        if sensor_index is None:
            sensor_fn = self._get_sensors_for_vis
        else:

            def sensor_fn(stated: StateDict) -> tuple[jax.Array, jax.Array]:
                return self._get_selected_sensor(stated, self._sensor_index)

        if partial_range_x is None:
            x_range = self._x_range
        else:
            x_range = partial_range_x

        if partial_range_y is None:
            y_range = self._y_range
        else:
            y_range = partial_range_y

        return moderngl_vis.MglVisualizer(
            x_range=x_range,
            y_range=y_range,
            space=self._physics,
            stated=state.physics,
            sc_color=self._food_color,
            head_color=np.array(HEAD_COLOR) / 255.0,
            figsize=figsize,
            backend=backend,
            sensor_fn=None if no_sensor else sensor_fn,  # type: ignore
            **kwargs,
        )
