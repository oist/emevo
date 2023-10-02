from emevo.env import Env
from typing import Literal, Callable


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
        food_mass: float = 0.25,
        food_friction: float = 0.1,
        food_initial_force: tuple[float, float] = (0.0, 0.0),
        foodloc_interval: int = 1000,
        wall_friction: float = 0.05,
        max_abs_impulse: float = 0.2,
        dt: float = 0.05,
        damping: float = 1.0,
        encount_threshold: int = 2,
        n_physics_steps: int = 5,
        max_place_attempts: int = 10,
        body_elasticity: float = 0.4,
        nofriction: bool = False,
    ) -> None:
        pass
