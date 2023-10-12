from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from emevo.env import Env
from emevo.environments.phyjax2d import Position, Space, State, StateDict
from emevo.environments.phyjax2d_utils import (
    SpaceBuilder,
    make_approx_circle,
    make_square,
)
from emevo.environments.utils.food_repr import ReprLoc, ReprLocFn, ReprNum, ReprNumFn
from emevo.environments.utils.locating import (
    CircleCoordinate,
    Coordinate,
    InitLoc,
    InitLocFn,
    SquareCoordinate,
)


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
        builder.add_circle(radius=food_radius, friction=0.0, elasticity=0.0)
    space = builder.build()
    return space, seg_state


class CircleForaging(Env):
    def __init__(
        self,
        n_initial_bodies: int = 6,
        food_num_fn: ReprNumFn | str | tuple[str, ...] = "constant",
        food_loc_fn: ReprLocFn | str | tuple[str, ...] = "gaussian",
        body_loc_fn: InitLocFn | str | tuple[str, ...] = "uniform",
        xlim: tuple[float, float] = (0.0, 200.0),
        ylim: tuple[float, float] = (0.0, 200.0),
        env_radius: float = 120.0,
        env_shape: Literal["square", "circle"] = "square",
        obstacles: list[tuple[float, float, float, float]] | None = None,
        n_agent_sensors: int = 8,
        sensor_length: float = 10.0,
        sensor_range: tuple[float, float] = (-180.0, 180.0),
        agent_radius: float = 12.0,
        agent_mass: float = 1.0,
        agent_friction: float = 0.1,
        food_radius: float = 4.0,
        food_mass: float = 0.1,
        food_friction: float = 0.0,
        foodloc_interval: int = 1000,
        max_abs_impulse: float = 0.2,
        dt: float = 0.05,
        damping: float = 1.0,
        n_physics_steps: int = 5,
        max_place_attempts: int = 10,
        body_elasticity: float = 0.4,
        nofriction: bool = False,
    ) -> None:
        if env_shape == "square":
            self._coordinate = SquareCoordinate(xlim, ylim)
        elif env_shape == "circle":
            self._coordinate = CircleCoordinate((env_radius, env_radius), env_radius)
        else:
            raise ValueError(f"Unsupported env_shape {env_shape}")
