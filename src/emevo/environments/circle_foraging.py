from __future__ import annotations

import enum
import functools
import warnings
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Callable, Literal, NamedTuple

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
    FoodNumState,
    Locating,
    LocatingFn,
    LocatingState,
    ReprNum,
    ReprNumFn,
    SquareCoordinate,
    first_true,
    loc_gaussian,
    place,
)
from emevo.environments.phyjax2d import Circle, Position, Raycast, ShapeDict
from emevo.environments.phyjax2d import Space as Physics
from emevo.environments.phyjax2d import (
    State,
    StateDict,
    Velocity,
    VelocitySolver,
    circle_raycast,
    segment_raycast,
)
from emevo.environments.phyjax2d import step as physics_step
from emevo.environments.phyjax2d_utils import (
    Color,
    SpaceBuilder,
    make_approx_circle,
    make_square,
)
from emevo.spaces import BoxSpace, NamedTupleSpace
from emevo.vec2d import Vec2d

MAX_ANGULAR_VELOCITY: float = float(np.pi)
MAX_VELOCITY: float = 10.0
AGENT_COLOR: Color = Color(2, 204, 254)
FOOD_COLOR: Color = Color(254, 2, 162)
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
                self.collision,
                self.velocity,
                jnp.expand_dims(self.angle, axis=1),
                jnp.expand_dims(self.angular_velocity, axis=1),
                jnp.expand_dims(self.energy, axis=1),
            ),
            axis=1,
        )


@chex.dataclass
class CFState:
    physics: StateDict
    solver: VelocitySolver
    food_num: FoodNumState
    agent_loc: LocatingState
    food_loc: LocatingState
    key: chex.PRNGKey
    step: jax.Array
    unique_id: UniqueID
    status: Status
    n_born_agents: jax.Array

    @property
    def stated(self) -> StateDict:
        return self.physics

    def is_extinct(self) -> bool:
        return jnp.logical_not(jnp.any(self.unique_id.is_active())).item()


class Obstacle(str, enum.Enum):
    NONE = "none"
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
        elif self == Obstacle.CENTER_HALF:
            return [(Vec2d(width / 2, height / 2), Vec2d(width / 2, height))]
        elif self == Obstacle.CENTER_TWO_THIRDS:
            return [(Vec2d(width / 2, height / 3), Vec2d(width / 2, height))]
        else:
            raise ValueError(f"Unsupported Obstacle: {self}")


class SensorRange(str, enum.Enum):
    NARROW = "narrow"
    WIDE = "wide"
    ALL = "all"

    def as_tuple(self) -> tuple[float, float]:
        if self == SensorRange.NARROW:
            return -30.0, 30.0
        elif self == SensorRange.WIDE:
            return -60.0, 60.0
        else:
            return -180.0, 180.0


def _get_num_or_loc_fn(
    arg: str | tuple | list | Callable[..., Any],
    enum_type: Callable[..., Callable[..., Any]],
    default_args: dict[str, tuple[Any, ...]],
) -> Any:
    if callable(arg):
        return arg
    if isinstance(arg, str):
        return enum_type(arg)(*default_args[arg])
    elif isinstance(arg, tuple) or isinstance(arg, list):
        name, *args = arg
        return enum_type(name)(*args)
    else:
        raise ValueError(f"Invalid value in _get_num_or_loc_fn {arg}")


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
            friction=0.1,
            elasticity=0.1,
            color=FOOD_COLOR,
            is_static=True,
        )
    return builder.build()


def _observe_closest(
    shaped: ShapeDict,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
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
        jax.tree_map(
            lambda arr: jnp.max(arr, keepdims=True),
            (to_c, to_sc, to_seg),
        ),
    )
    return jnp.where(obs == jnp.max(obs, axis=-1, keepdims=True), obs, -1.0)


_vmap_obs = jax.vmap(_observe_closest, in_axes=(None, 0, 0, None))


def _get_sensors(
    shaped: ShapeDict,
    n_sensors: int,
    sensor_range: tuple[float, float],
    sensor_length: float,
    stated: StateDict,
) -> tuple[jax.Array, jax.Array]:
    assert shaped.circle is not None and stated.circle is not None
    radius = shaped.circle.radius
    p1 = jnp.stack((jnp.zeros_like(radius), radius), axis=1)  # (N, 2)
    p1 = jnp.repeat(p1, n_sensors, axis=0)  # (N x M, 2)
    p2 = p1 + jnp.array([0.0, sensor_length])  # (N x M, 2)
    sensor_rad = jnp.deg2rad(jnp.linspace(*sensor_range, n_sensors))
    sensor_p = Position(
        angle=jax.vmap(lambda x: x + sensor_rad)(stated.circle.p.angle).ravel(),
        xy=jnp.repeat(stated.circle.p.xy, n_sensors, axis=0),
    )
    p1 = sensor_p.transform(p1)
    p2 = sensor_p.transform(p2)
    return p1, p2


def get_sensor_obs(
    shaped: ShapeDict,
    n_sensors: int,
    sensor_range: tuple[float, float],
    sensor_length: float,
    stated: StateDict,
) -> jax.Array:
    assert stated.circle is not None
    p1, p2 = _get_sensors(shaped, n_sensors, sensor_range, sensor_length, stated)
    return _vmap_obs(shaped, p1, p2, stated)


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


def _first_n_true(boolean_array: jax.Array, n: jax.Array) -> jax.Array:
    return jnp.logical_and(boolean_array, jnp.cumsum(boolean_array) <= n)


def _nonzero(arr: jax.Array, n: int) -> jax.Array:
    cums = jnp.cumsum(arr)
    bincount = jnp.zeros(n, dtype=jnp.int32).at[cums].add(1)
    return jnp.cumsum(bincount)


class CircleForaging(Env):
    def __init__(
        self,
        n_initial_agents: int = 6,
        n_max_agents: int = 100,
        n_max_foods: int = 40,
        food_num_fn: ReprNumFn | str | tuple[str, ...] = "constant",
        food_loc_fn: LocatingFn | str | tuple[str, ...] = "gaussian",
        agent_loc_fn: LocatingFn | str | tuple[str, ...] = "uniform",
        xlim: tuple[float, float] = (0.0, 200.0),
        ylim: tuple[float, float] = (0.0, 200.0),
        env_radius: float = 120.0,
        env_shape: Literal["square", "circle"] = "square",
        obstacles: list[tuple[Vec2d, Vec2d]] | str = "none",
        newborn_loc: Literal["neighbor", "uniform"] = "neighbor",
        neighbor_stddev: float = 40.0,
        n_agent_sensors: int = 16,
        sensor_length: float = 100.0,
        sensor_range: tuple[float, float] | SensorRange = SensorRange.WIDE,
        agent_radius: float = 10.0,
        food_radius: float = 4.0,
        foodloc_interval: int = 1000,
        dt: float = 0.1,
        linear_damping: float = 0.8,
        angular_damping: float = 0.6,
        max_force: float = 40.0,
        min_force: float = -20.0,
        init_energy: float = 20.0,
        energy_capacity: float = 100.0,
        force_energy_consumption: float = 0.01 / 40.0,
        basic_energy_consumption: float = 0.0,
        energy_share_ratio: float = 0.4,
        n_velocity_iter: int = 6,
        n_position_iter: int = 2,
        n_physics_iter: int = 5,
        max_place_attempts: int = 10,
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
        self._food_loc_fn, self._initial_foodloc_state = self._make_food_loc_fn(
            food_loc_fn
        )
        self._food_num_fn, self._initial_foodnum_state = self._make_food_num_fn(
            food_num_fn
        )
        self._agent_loc_fn, self._initial_agentloc_state = self._make_agent_loc_fn(
            agent_loc_fn
        )
        # Energy
        self._force_energy_consumption = force_energy_consumption
        self._basic_energy_consumption = basic_energy_consumption
        self._init_energy = init_energy
        self._energy_capacity = energy_capacity
        self._energy_share_ratio = energy_share_ratio
        # Initial numbers
        assert n_max_agents > n_initial_agents
        assert n_max_foods > self._food_num_fn.initial
        self._n_initial_agents = n_initial_agents
        self.n_max_agents = n_max_agents
        self._n_initial_foods = self._food_num_fn.initial
        self._n_max_foods = n_max_foods
        self._max_place_attempts = max_place_attempts
        # Physics
        if isinstance(obstacles, str):
            obs_list = Obstacle(obstacles).as_list(self._x_range, self._y_range)
        else:
            obs_list = obstacles

        self._physics = _make_physics(
            dt=dt,
            coordinate=self._coordinate,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
            n_velocity_iter=n_velocity_iter,
            n_position_iter=n_position_iter,
            n_max_agents=n_max_agents,
            n_max_foods=n_max_foods,
            agent_radius=agent_radius,
            food_radius=food_radius,
            obstacles=obs_list,
        )
        self._agent_indices = jnp.arange(n_max_agents)
        self._food_indices = jnp.arange(n_max_foods)
        self._n_physics_iter = n_physics_iter
        # Spaces
        self.act_space = BoxSpace(low=min_force, high=max_force, shape=(2,))
        self.obs_space = NamedTupleSpace(
            CFObs,
            sensor=BoxSpace(low=0.0, high=1.0, shape=(n_agent_sensors, N_OBJECTS)),
            collision=BoxSpace(low=0.0, high=1.0, shape=(N_OBJECTS,)),
            velocity=BoxSpace(low=-MAX_VELOCITY, high=MAX_VELOCITY, shape=(2,)),
            angle=BoxSpace(low=-2 * np.pi, high=2 * np.pi, shape=()),
            angular_velocity=BoxSpace(low=-np.pi / 10, high=np.pi / 10, shape=()),
            energy=BoxSpace(low=0.0, high=energy_capacity, shape=()),
        )
        # Obs
        self._n_sensors = n_agent_sensors
        # Some cached constants
        act_p1 = Vec2d(0, agent_radius).rotated(np.pi * 0.75)
        act_p2 = Vec2d(0, agent_radius).rotated(-np.pi * 0.75)
        self._act_p1 = jnp.tile(jnp.array(act_p1), (self.n_max_agents, 1))
        self._act_p2 = jnp.tile(jnp.array(act_p2), (self.n_max_agents, 1))
        self._init_agent = jax.jit(
            functools.partial(
                place,
                n_trial=self._max_place_attempts,
                radius=self._agent_radius,
                coordinate=self._coordinate,
                loc_fn=self._agent_loc_fn,
                shaped=self._physics.shaped,
            )
        )
        self._place_food = jax.jit(
            functools.partial(
                place,
                n_trial=self._max_place_attempts,
                radius=self._food_radius,
                coordinate=self._coordinate,
                loc_fn=self._food_loc_fn,
                shaped=self._physics.shaped,
            )
        )
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
                loc_fn = loc_gaussian(
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
                    stated=stated,
                )

            self._place_newborn = jax.vmap(
                place_newborn_neighbor,
                in_axes=(None, None, 0, 0),
            )
        else:
            raise ValueError(f"Invalid newborn_loc: {newborn_loc}")
        if isinstance(sensor_range, SensorRange):
            sensor_range_tuple = SensorRange(sensor_range).as_tuple()
        else:
            sensor_range_tuple = sensor_range
        self._sensor_obs = jax.jit(
            functools.partial(
                get_sensor_obs,
                shaped=self._physics.shaped,
                n_sensors=n_agent_sensors,
                sensor_range=sensor_range_tuple,
                sensor_length=sensor_length,
            )
        )

        # For visualization
        self._get_sensors = jax.jit(
            functools.partial(
                _get_sensors,
                shaped=self._physics.shaped,
                n_sensors=n_agent_sensors,
                sensor_range=sensor_range_tuple,
                sensor_length=sensor_length,
            )
        )

    @staticmethod
    def _make_food_num_fn(
        food_num_fn: str | tuple | ReprNumFn,
    ) -> tuple[ReprNumFn, FoodNumState]:
        return _get_num_or_loc_fn(
            food_num_fn,
            ReprNum,  # type: ignore
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

    def step(
        self,
        state: CFState,
        action: ArrayLike,
    ) -> tuple[CFState, TimeStep[CFObs]]:
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
        # This is also used in computing energy_delta
        food_collision = jnp.max(c2sc, axis=1)
        collision = jnp.stack(
            (jnp.max(c2c, axis=1), food_collision, jnp.max(seg2c, axis=0)),
            axis=1,
        )
        # Gather sensor obs
        sensor_obs = self._sensor_obs(stated=stated)
        # energy_delta = food - coef * |force|
        force_norm = jnp.sqrt(f1_raw**2 + f2_raw**2).ravel()
        energy_delta = (
            food_collision
            - self._force_energy_consumption * force_norm
            - self._basic_energy_consumption
        )
        # Remove and reproduce foods
        key, food_key = jax.random.split(state.key)
        stated, food_num, food_loc = self._remove_and_reproduce_foods(
            food_key,
            jnp.max(c2sc, axis=0),
            stated,
            state.food_num,
            state.food_loc,
        )
        status = state.status.update(
            energy_delta=energy_delta,
            capacity=self._energy_capacity,
        )
        # Construct obs
        obs = CFObs(
            sensor=sensor_obs.reshape(-1, self._n_sensors, 3),
            collision=collision,
            angle=stated.circle.p.angle,
            velocity=stated.circle.v.xy,
            angular_velocity=stated.circle.v.angle,
            energy=status.energy,
        )
        timestep = TimeStep(encount=c2c, obs=obs)
        state = CFState(
            physics=stated,
            solver=solver,
            food_num=food_num,
            agent_loc=state.agent_loc,
            food_loc=food_loc,
            key=key,
            step=state.step + 1,
            unique_id=state.unique_id,
            status=status.step(),
            n_born_agents=state.n_born_agents,
        )
        return state, timestep

    def activate(
        self,
        state: CFState,
        is_parent: jax.Array,
    ) -> tuple[CFState, jax.Array]:
        N = self.n_max_agents
        circle = state.physics.circle
        keys = jax.random.split(state.key, N + 1)
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
        parent_indices = _nonzero(is_parent, N)
        # empty_indices := nonzero_indices(not(is_active)) + (N, N, N, ....)
        replaced_indices = _nonzero(is_replaced, N)
        # To use .at[].add, append (0, 0) to sampled xy
        new_xy_with_sentinel = jnp.concatenate((new_xy, jnp.zeros((1, 2))))
        xy = circle.p.xy.at[replaced_indices].add(new_xy_with_sentinel[parent_indices])
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
        n_children = jnp.sum(is_parent)
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

    def deactivate(self, state: CFState, flag: jax.Array) -> CFState:
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

    def reset(self, key: chex.PRNGKey) -> tuple[CFState, TimeStep[CFObs]]:
        physics, agent_loc, food_loc = self._initialize_physics_state(key)
        N = self.n_max_agents
        unique_id = init_uniqueid(self._n_initial_agents, N)
        status = init_status(N, self._init_energy)
        state = CFState(
            physics=physics,
            solver=self._physics.init_solver(),
            agent_loc=agent_loc,
            food_loc=food_loc,
            food_num=self._initial_foodnum_state,
            key=key,
            step=jnp.array(0, dtype=jnp.int32),
            unique_id=unique_id,
            status=status,
            n_born_agents=jnp.array(self._n_initial_agents, dtype=jnp.int32),
        )
        sensor_obs = self._sensor_obs(stated=physics)
        obs = CFObs(
            sensor=sensor_obs.reshape(-1, self._n_sensors, N_OBJECTS),
            collision=jnp.zeros((N, N_OBJECTS), dtype=bool),
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
    ) -> tuple[StateDict, LocatingState, LocatingState]:
        # Set segment
        stated = self._physics.shaped.zeros_state()
        assert stated.circle is not None

        # Set is_active
        is_active_c = jnp.concatenate(
            (
                jnp.ones(self._n_initial_agents, dtype=bool),
                jnp.zeros(self.n_max_agents - self._n_initial_agents, dtype=bool),
            )
        )
        is_active_s = jnp.concatenate(
            (
                jnp.ones(self._n_initial_foods, dtype=bool),
                jnp.zeros(self._n_max_foods - self._n_initial_foods, dtype=bool),
            )
        )
        stated = stated.nested_replace("circle.is_active", is_active_c)
        stated = stated.nested_replace("static_circle.is_active", is_active_s)
        # Move all circle to the invisiable area
        stated = stated.nested_replace(
            "circle.p.xy",
            jnp.ones_like(stated.circle.p.xy) * NOWHERE,
        )
        stated = stated.nested_replace(
            "static_circle.p.xy",
            jnp.ones_like(stated.static_circle.p.xy) * NOWHERE,
        )
        keys = jax.random.split(key, self._n_initial_agents + self._n_initial_foods)
        agent_failed = 0
        agentloc_state = self._initial_foodloc_state
        for i, key in enumerate(keys[: self._n_initial_agents]):
            xy, ok = self._init_agent(loc_state=agentloc_state, key=key, stated=stated)
            if ok:
                stated = stated.nested_replace(
                    "circle.p.xy",
                    stated.circle.p.xy.at[i].set(xy),
                )
                agentloc_state = agentloc_state.increment()
            else:
                agent_failed += 1

        if agent_failed > 0:
            warnings.warn(f"Failed to place {agent_failed} agents!", stacklevel=1)

        food_failed = 0
        foodloc_state = self._initial_foodloc_state
        for i, key in enumerate(keys[self._n_initial_agents :]):
            xy, ok = self._place_food(loc_state=foodloc_state, key=key, stated=stated)
            if ok:
                stated = stated.nested_replace(
                    "static_circle.p.xy",
                    stated.static_circle.p.xy.at[i].set(xy),
                )
                foodloc_state = foodloc_state.increment()
            else:
                food_failed += 1

        if food_failed > 0:
            warnings.warn(f"Failed to place {food_failed} foods!", stacklevel=1)

        return stated, agentloc_state, foodloc_state

    def _remove_and_reproduce_foods(
        self,
        key: chex.PRNGKey,
        eaten: jax.Array,
        sd: StateDict,
        food_num: FoodNumState,
        food_loc: LocatingState,
    ) -> tuple[StateDict, FoodNumState, LocatingState]:
        # Remove foods
        xy = jnp.where(
            jnp.expand_dims(eaten, axis=1),
            jnp.ones_like(sd.static_circle.p.xy) * NOWHERE,
            sd.static_circle.p.xy,
        )
        is_active = jnp.logical_and(sd.static_circle.is_active, jnp.logical_not(eaten))
        food_num = self._food_num_fn(food_num.eaten(jnp.sum(eaten)))
        # Generate new foods
        first_inactive = first_true(jnp.logical_not(is_active))
        new_food, ok = self._place_food(loc_state=food_loc, key=key, stated=sd)
        place = jnp.logical_and(jnp.logical_and(ok, food_num.appears()), first_inactive)
        xy = jnp.where(
            jnp.expand_dims(place, axis=1),
            jnp.expand_dims(new_food, axis=0),
            xy,
        )
        is_active = jnp.logical_or(is_active, place)
        p = replace(sd.static_circle.p, xy=xy)
        sc = replace(sd.static_circle, p=p, is_active=is_active)
        sd = replace(sd, static_circle=sc)
        incr = jnp.sum(place)
        return sd, food_num.recover(incr), food_loc.increment(incr)

    def visualizer(
        self,
        state: CFState,
        figsize: tuple[float, float] | None = None,
        backend: str = "pyglet",
        **kwargs,
    ) -> Visualizer[StateDict]:
        """Create a visualizer for the environment"""
        from emevo.environments import moderngl_vis

        return moderngl_vis.MglVisualizer(
            x_range=self._x_range,
            y_range=self._y_range,
            space=self._physics,
            stated=state.physics,
            figsize=figsize,
            backend=backend,
            sensor_fn=self._get_sensors,
            **kwargs,
        )
