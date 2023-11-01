from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from emevo.env import Env, Profile, TimeStep, Visualizer, init_profile
from emevo.environments.locating import (
    CircleCoordinate,
    Locating,
    LocatingFn,
    LocatingState,
    SquareCoordinate,
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
from emevo.environments.placement import place
from emevo.environments.reproduction import FoodNumState, ReprNum, ReprNumFn
from emevo.spaces import BoxSpace, NamedTupleSpace
from emevo.types import Index
from emevo.vec2d import Vec2d

MAX_ANGULAR_VELOCITY: float = float(np.pi)
MAX_VELOCITY: float = 10.0
MAX_FORCE: float = 1.0
AGENT_COLOR: Color = Color(2, 204, 254)
FOOD_COLOR: Color = Color(254, 2, 162)


class CFObs(NamedTuple):
    """Observation of an agent."""

    sensor: jax.Array
    collision: jax.Array
    velocity: jax.Array
    angle: jax.Array
    angular_velocity: jax.Array
    energy: jax.Array

    def __array__(self) -> jax.Array:
        return jnp.concatenate(
            (
                self.sensor.ravel(),
                self.collision.ravel(),
                self.velocity,
                self.angle,
                self.angular_velocity,
                self.energy,
            )
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
    profile: Profile
    n_born_agents: jax.Array

    @property
    def stated(self) -> StateDict:
        return self.physics

    def is_extinct(self) -> bool:
        return jnp.logical_not(jnp.any(self.profile.is_active())).item()


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
    linear_damping: float = 0.9,
    angular_damping: float = 0.9,
    n_velocity_iter: int = 6,
    n_position_iter: int = 2,
    n_max_agents: int = 40,
    n_max_foods: int = 20,
    agent_radius: float = 10.0,
    food_radius: float = 4.0,
) -> tuple[Physics, State]:
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
        outer_walls = make_approx_circle(coordinate.center, coordinate.radius)
    else:
        outer_walls = make_square(
            *coordinate.xlim,
            *coordinate.ylim,
            rounded_offset=np.floor(food_radius * 2 / (np.sqrt(2) - 1.0)),
        )
    segments = []
    for wall in outer_walls:
        a2b = wall[1] - wall[0]
        angle = jnp.array(a2b.angle)
        xy = jnp.array(wall[0] + wall[1]) / 2
        position = Position(angle=angle, xy=xy)
        segments.append(position)
        builder.add_segment(length=a2b.length, friction=0.1, elasticity=0.2)
    seg_position = jax.tree_map(lambda *args: jnp.stack(args), *segments)
    seg_state = State.from_position(seg_position)
    for _ in range(n_max_agents):
        builder.add_circle(
            radius=agent_radius,
            friction=0.1,
            elasticity=0.2,
            density=0.04,
            color=AGENT_COLOR,
        )
    for _ in range(n_max_foods):
        builder.add_circle(
            radius=food_radius,
            friction=0.1,
            elasticity=0.1,
            color=FOOD_COLOR,
            is_static=True,
        )
    return builder.build(), seg_state


def _observe_closest(
    shaped: ShapeDict,
    p1: jax.Array,
    p2: jax.Array,
    stated: StateDict,
) -> jax.Array:
    assert shaped.circle is not None and stated.circle is not None
    assert shaped.static_circle is not None and stated.static_circle is not None
    assert shaped.segment is not None and stated.segment is not None

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


@functools.partial(jax.jit, static_argnums=(0, 1))
def nstep(
    n: int,
    space: Physics,
    stated: StateDict,
    solver: VelocitySolver,
) -> tuple[StateDict, VelocitySolver, jax.Array]:
    def body(
        stated_and_solver: tuple[StateDict, VelocitySolver],
        _zero: jax.Array,
    ) -> tuple[tuple[StateDict, VelocitySolver], jax.Array]:
        state, solver, contact = physics_step(space, *stated_and_solver)
        return (state, solver), contact.penetration >= 0.0

    (state, solver), contacts = jax.lax.scan(body, (stated, solver), jnp.zeros(n))
    return state, solver, contacts


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
        obstacles: list[tuple[float, float, float, float]] | None = None,
        n_agent_sensors: int = 8,
        sensor_length: float = 10.0,
        sensor_range: tuple[float, float] = (-120.0, 120.0),
        agent_radius: float = 10.0,
        food_radius: float = 4.0,
        foodloc_interval: int = 1000,
        dt: float = 0.1,
        linear_damping: float = 0.9,
        angular_damping: float = 0.8,
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
        # Initial numbers
        assert n_max_agents > n_initial_agents
        assert n_max_foods > self._food_num_fn.initial
        self._n_initial_agents = n_initial_agents
        self._n_max_agents = n_max_agents
        self._n_initial_foods = self._food_num_fn.initial
        self._n_max_foods = n_max_foods
        self._max_place_attempts = max_place_attempts
        # Physics
        self._physics, self._segment_state = _make_physics(
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
        )
        self._agent_indices = jnp.arange(n_max_agents)
        self._food_indices = jnp.arange(n_max_foods)
        self._n_physics_iter = n_physics_iter
        # Spaces
        N = self._n_max_agents
        self.act_space = BoxSpace(low=0.0, high=MAX_FORCE, shape=(N, 2))
        self.obs_space = NamedTupleSpace(
            CFObs,
            sensor=BoxSpace(low=0.0, high=1.0, shape=(N, n_agent_sensors, 3)),
            collision=BoxSpace(low=0.0, high=1.0, shape=(N, 3)),
            velocity=BoxSpace(low=-MAX_VELOCITY, high=MAX_VELOCITY, shape=(N, 2)),
            angle=BoxSpace(low=-2 * np.pi, high=2 * np.pi, shape=(N,)),
            angular_velocity=BoxSpace(low=-np.pi / 10, high=np.pi / 10, shape=(N,)),
            energy=BoxSpace(low=0.0, high=50.0, shape=(N,)),
        )
        # Some cached constants
        self._invisible_xy = jnp.array([-100.0, -100.0], dtype=jnp.float32)
        act_p1 = Vec2d(0, agent_radius).rotated(np.pi * 0.75)
        act_p2 = Vec2d(0, agent_radius).rotated(-np.pi * 0.75)
        N = self._n_max_agents
        self._act_p1 = jnp.tile(jnp.array(act_p1), (N, 1))
        self._act_p2 = jnp.tile(jnp.array(act_p2), (N, 1))
        self._place_agent = jax.jit(
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

    @staticmethod
    def _make_food_num_fn(
        food_num_fn: str | tuple | ReprNumFn,
    ) -> tuple[ReprNumFn, FoodNumState]:
        return _get_num_or_loc_fn(
            food_num_fn,
            ReprNum,  # type: ignore
            {"constant": (10,), "linear": (10, 0.01), "logistic": (8, 1.2, 12)},
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

    def set_food_num_fn(self, food_num_fn: str | tuple | ReprNumFn) -> None:
        self._food_num_fn = self._make_food_num_fn(food_num_fn)

    def set_food_loc_fn(self, food_loc_fn: str | tuple | LocatingFn) -> None:
        self._food_loc_fn = self._make_food_loc_fn(food_loc_fn)

    def set_agent_loc_fn(self, agent_loc_fn: str | tuple | LocatingFn) -> None:
        self._agent_loc_fn = self._make_agent_loc_fn(agent_loc_fn)

    def step(self, state: CFState, action: ArrayLike) -> CFState:
        # Add force
        act = self.act_space.clip(jnp.array(action))
        f1, f2 = act[:, 0], act[:, 1]
        f1 = jnp.stack((jnp.zeros_like(f1), f1), axis=1) * -self._act_p1
        f2 = jnp.stack((jnp.zeros_like(f2), f2), axis=1) * -self._act_p2
        circle = state.physics.circle
        circle = circle.apply_force_local(self._act_p1, f1)
        circle = circle.apply_force_local(self._act_p2, f2)
        stated = state.physics.replace(circle=circle)
        # Step physics simulator
        stated, solver, contacts = nstep(
            self._n_physics_iter,
            self._physics,
            stated,
            state.solver,
        )
        circle_contacts = self._physics.get_specific_contact(
            "circle",
            jnp.max(contacts, axis=0),
        )
        return state.replace(physics=stated, solver=solver)

    def activate(self, state: CFState, parent_gen: jax.Array) -> tuple[CFState, bool]:
        key, activate_key = jax.random.split(state.key)
        (index,) = jnp.nonzero(
            jnp.logical_not(state.profile.is_active()),
            size=1,
            fill_value=-1,
        )
        index = index[0]
        xy = self._place_agent(key=activate_key, stated=state.physics)
        ok = jnp.logical_and(index >= 0, jnp.all(xy < jnp.inf))

        def success() -> tuple[CFState, bool]:
            circle_xy = state.physics.circle.p.xy.at[index].set(xy)
            circle_angle = state.physics.circle.p.angle.at[index].set(0.0)
            p = Position(angle=circle_angle, xy=circle_xy)
            is_active = state.physics.circle.is_active.at[index].set(True)
            circle = state.physics.circle.replace(p=p, is_active=is_active)
            physics = state.physics.replace(circle=circle)
            profile = state.profile.activate(
                index,
                parent_gen,
                state.n_born_agents,
                state.step,
            )
            new_state = state.replace(
                physics=physics,
                profile=profile,
                n_born_agents=state.n_born_agents + 1,
                key=key,
            )
            return new_state, True

        def failure() -> tuple[CFState, bool]:
            return state.replace(key=key), False

        return jax.lax.cond(ok, success, failure)

    def deactivate(self, state: CFState, index: Index) -> tuple[CFState, bool]:
        ok = state.profile.is_active()[index]

        def success() -> tuple[CFState, bool]:
            p_xy = state.physics.circle.p.xy.at[index].set(self._invisible_xy)
            p = state.physics.circle.p.replace(xy=p_xy)
            v_xy = state.physics.circle.v.xy.at[index].set(jnp.zeros(2))
            v_angle = state.physics.circle.v.angle.at[index].set(0)
            v = Velocity(angle=v_angle, xy=v_xy)
            is_active = state.physics.circle.is_active.at[index].set(False)
            circle = state.physics.circle.replace(p=p, v=v, is_active=is_active)
            physics = state.physics.replace(circle=circle)
            profile = state.profile.deactivate(index)
            return state.replace(physics=physics, profile=profile), True

        return jax.lax.cond(ok, success, lambda: (state, False))

    def reset(self, key: chex.PRNGKey) -> CFState:
        state_key, init_key = jax.random.split(key)
        physics, agent_loc, food_loc = self._initialize_physics_state(init_key)
        return CFState(  # type: ignore
            physics=physics,
            solver=self._physics.init_solver(),
            agent_loc=agent_loc,
            food_loc=food_loc,
            food_num=self._initial_foodnum_state,
            # Protocols
            key=state_key,
            step=jnp.array(0, dtype=jnp.int32),
            profile=init_profile(self._n_initial_agents, self._n_max_agents),
            n_born_agents=jnp.array(self._n_initial_agents, dtype=jnp.int32),
        )

    def _initialize_physics_state(
        self,
        key: chex.PRNGKey,
    ) -> tuple[StateDict, LocatingState, LocatingState]:
        stated = self._physics.shaped.zeros_state()
        assert stated.circle is not None

        # Set is_active
        is_active_c = jnp.concatenate(
            (
                jnp.ones(self._n_initial_agents, dtype=bool),
                jnp.zeros(self._n_max_agents - self._n_initial_agents, dtype=bool),
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
            jnp.ones_like(stated.circle.p.xy) * -100,
        )
        stated = stated.nested_replace(
            "static_circle.p.xy",
            jnp.ones_like(stated.static_circle.p.xy) * -100,
        )
        keys = jax.random.split(key, self._n_initial_agents + self._n_initial_foods)
        agent_failed = 0
        agentloc_state = self._initial_foodloc_state
        for i, key in enumerate(keys[: self._n_initial_agents]):
            xy = self._place_agent(loc_state=agentloc_state, key=key, stated=stated)
            if jnp.all(xy < jnp.inf):
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
            xy = self._place_food(loc_state=foodloc_state, key=key, stated=stated)
            if jnp.all(xy < jnp.inf):
                stated = stated.nested_replace(
                    "static_circle.p.xy",
                    stated.static_circle.p.xy.at[i].set(xy),
                )
                foodloc_state = foodloc_state.increment()
            else:
                food_failed += 1

        if food_failed > 0:
            warnings.warn(f"Failed to place {food_failed} foods!", stacklevel=1)

        stated = stated.replace(segment=self._segment_state)
        return stated, agentloc_state, foodloc_state

    def visualizer(
        self,
        state: CFState,
        figsize: tuple[float, float] | None = None,
        mgl_backend: str = "pyglet",
        **kwargs,
    ) -> Visualizer:
        """Create a visualizer for the environment"""
        from emevo.environments import moderngl_vis

        return moderngl_vis.MglVisualizer(
            x_range=self._x_range,
            y_range=self._y_range,
            space=self._physics,
            stated=state.physics,
            figsize=figsize,
            backend=mgl_backend,
            **kwargs,
        )
