import warnings
from typing import Any, Callable, Literal, NamedTuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from emevo.env import Env, Profile, Visualizer
from emevo.environments.phyjax2d import Position, Space, State, StateDict
from emevo.environments.phyjax2d_utils import (
    SpaceBuilder,
    make_approx_circle,
    make_square,
)
from emevo.environments.placement import _inf_xy, place_agent, place_food
from emevo.environments.utils.food_repr import (
    FoodNumState,
    ReprLoc,
    ReprLocFn,
    ReprLocState,
    ReprNum,
    ReprNumFn,
)
from emevo.environments.utils.locating import (
    CircleCoordinate,
    InitLoc,
    InitLocFn,
    SquareCoordinate,
)
from emevo.types import Index

FN = TypeVar("FN")


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
                self.collision,
                self.velocity,
                self.angle,
                self.angular_velocity,
                self.energy,
            )
        )


@chex.dataclass
class CFState:
    physics: StateDict
    food_num: FoodNumState
    repr_loc: ReprLocState


def _get_num_or_loc_fn(
    arg: str | tuple | FN,
    enum_type: Callable[..., Callable[..., Any]],
    default_args: dict[str, tuple[Any, ...]],
) -> Any:
    if isinstance(arg, str):
        return enum_type(arg)(*default_args[arg])
    elif isinstance(arg, tuple) or isinstance(arg, list):
        name, *args = arg
        return enum_type(name)(*args)
    else:
        return arg


def _make_space(
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
) -> tuple[Space, State]:
    builder = SpaceBuilder(
        gravity=(0.0, 0.0),  # No gravity
        dt=dt,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        n_velocity_iter=n_velocity_iter,
        n_position_iter=n_position_iter,
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
        # Use the default density for now
        builder.add_circle(radius=agent_radius, friction=0.1, elasticity=0.2)
    for _ in range(n_max_foods):
        builder.add_circle(radius=food_radius, friction=0.0, elasticity=0.2)
    space = builder.build()
    return space, seg_state


class CircleForaging(Env):
    def __init__(
        self,
        n_initial_agents: int = 6,
        n_max_agents: int = 100,
        n_max_foods: int = 100,
        food_num_fn: ReprNumFn | str | tuple[str, ...] = "constant",
        food_loc_fn: ReprLocFn | str | tuple[str, ...] = "gaussian",
        agent_loc_fn: InitLocFn | str | tuple[str, ...] = "uniform",
        xlim: tuple[float, float] = (0.0, 200.0),
        ylim: tuple[float, float] = (0.0, 200.0),
        env_radius: float = 120.0,
        env_shape: Literal["square", "circle"] = "square",
        obstacles: list[tuple[float, float, float, float]] | None = None,
        n_agent_sensors: int = 8,
        sensor_length: float = 10.0,
        sensor_range: tuple[float, float] = (-180.0, 180.0),
        agent_radius: float = 12.0,
        food_radius: float = 4.0,
        foodloc_interval: int = 1000,
        max_abs_impulse: float = 0.2,
        dt: float = 0.05,
        linear_damping: float = 0.9,
        angular_damping: float = 0.8,
        n_velocity_iter: int = 6,
        n_position_iter: int = 2,
        n_physics_steps: int = 5,
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
        self._agent_loc_fn = self._make_agent_loc_fn(agent_loc_fn)
        # Initial numbers
        assert n_max_agents > n_initial_agents
        assert n_max_foods > self._food_num_fn.initial
        self._n_initial_agents = n_initial_agents
        self._n_max_agents = n_max_agents
        self._n_initial_foods = self._food_num_fn.initial
        self._n_max_foods = n_max_foods
        self._max_place_attempts = max_place_attempts
        # Physics
        self._space, self._segment_state = _make_space(
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
        self._n_physics_steps = n_physics_steps

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
        food_loc_fn: str | tuple | ReprLocFn,
    ) -> tuple[ReprLocFn, ReprLocState]:
        return _get_num_or_loc_fn(
            food_loc_fn,
            ReprLoc,  # type: ignore
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

    def _make_agent_loc_fn(self, init_loc_fn: str | tuple | InitLocFn) -> InitLocFn:
        return _get_num_or_loc_fn(
            init_loc_fn,
            InitLoc,  # type: ignore
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

    def set_food_loc_fn(self, food_loc_fn: str | tuple | ReprLocFn) -> None:
        self._food_loc_fn = self._make_food_loc_fn(food_loc_fn)

    def set_agent_loc_fn(self, agent_loc_fn: str | tuple | InitLocFn) -> None:
        self._agent_loc_fn = self._make_agent_loc_fn(agent_loc_fn)

    def step(self, state: CFState, action: ArrayLike):
        pass

    def activate(self, state: CFState, index: Index) -> CFState:
        pass

    def deactivate(self, state: CFState, index: Index) -> CFState:
        pass

    def is_extinct(self, state: CFState) -> bool:
        pass

    def profile(self) -> Profile:
        pass

    def reset(self, key: chex.PRNGKey) -> CFState:
        stated = self._initialize_physics_state(key)
        repr_loc = self._initial_foodloc_state
        food_num = self._initial_foodnum_state
        return CFState(physics=stated, repr_loc=repr_loc, food_num=food_num)

    def _initialize_physics_state(self, key: chex.PRNGKey) -> StateDict:
        stated = self._space.shaped.zeros_state()
        circle = stated.circle
        assert circle is not None
        circle_xy = circle.p.xy

        circle_isactive = jnp.concatenate(
            (
                jnp.ones(self._n_initial_agents, dtype=bool),
                jnp.zeros(self._n_max_agents - self._n_initial_agents, dtype=bool),
                jnp.ones(self._n_initial_foods, dtype=bool),
                jnp.zeros(self._n_max_foods - self._n_initial_foods, dtype=bool),
            )
        )
        stated = stated.replace(circle=stated.circle.replace(is_active=circle_isactive))  # type: ignore
        keys = jax.random.split(key, self._n_initial_foods + self._n_initial_agents)
        agent_failed = 0
        for i, key in enumerate(keys[: self._n_initial_foods]):
            xy = place_agent(
                n_trial=self._max_place_attempts,
                agent_radius=self._agent_radius,
                coordinate=self._coordinate,
                initloc_fn=self._agent_loc_fn,
                key=key,
                shaped=self._space.shaped,
                stated=stated,
            )
            if jnp.all(xy < _inf_xy):
                circle_xy = circle_xy.at[i].set(xy)
                circle = circle.replace(p=circle.p.replace(xy=circle_xy))  # type: ignore
            else:
                agent_failed += 1

        if agent_failed > 0:
            warnings.warn(f"Failed to place {agent_failed} agents!", stacklevel=1)

        food_failed = 0
        foodloc_state = self._initial_foodloc_state
        for i, key in enumerate(keys[self._n_initial_foods :]):
            xy = place_food(
                n_trial=self._max_place_attempts,
                food_radius=self._food_radius,
                coordinate=self._coordinate,
                reprloc_fn=self._food_loc_fn,  # type: ignore
                reprloc_state=foodloc_state,
                key=key,
                shaped=self._space.shaped,
                stated=stated,
            )
            if jnp.all(xy < _inf_xy):
                circle_xy = circle_xy.at[i + self._n_max_agents].set(xy)
                circle = circle.replace(p=circle.p.replace(xy=circle_xy))  # type: ignore
            else:
                food_failed += 1

        if food_failed > 0:
            warnings.warn(f"Failed to place {food_failed} foods!", stacklevel=1)

        return stated.replace(circle=circle, segment=self._segment_state)

    def visualizer(
        self,
        state: CFState,
        headless: bool = False,
        **kwargs,
    ) -> Visualizer:
        """Create a visualizer for the environment"""
        from emevo.environments.pymunk_envs import moderngl_vis

        return moderngl_vis.MglVisualizer(
            x_range=self._x_range,
            y_range=self._y_range,
            space=self._space,
            stated=state.physics,
            figsize=figsize,
            backend=mgl_backend,
            **kwargs,
        )