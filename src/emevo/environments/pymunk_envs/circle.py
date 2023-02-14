from __future__ import annotations

from functools import cached_property, partial
from typing import Any, Callable, Literal, NamedTuple, TypeVar

import numpy as np
import pymunk
from loguru import logger
from numpy.random import PCG64, Generator
from numpy.typing import NDArray
from pymunk.vec2d import Vec2d

from emevo.body import Body, Encount
from emevo.env import Env, Visualizer
from emevo.environments.pymunk_envs import pymunk_utils as utils
from emevo.environments.utils.color import Color
from emevo.environments.utils.food_repr import ReprLoc, ReprLocFn, ReprNum, ReprNumFn
from emevo.environments.utils.locating import (
    CircleCoordinate,
    Coordinate,
    InitLoc,
    InitLocFn,
    SquareCoordinate,
)
from emevo.spaces import BoxSpace, NamedTupleSpace

FN = TypeVar("FN")


class CFObs(NamedTuple):
    """Observation of an agent."""

    sensor: NDArray
    collision: NDArray
    velocity: NDArray
    angle: float
    energy: float

    def __array__(self) -> NDArray:
        return np.concatenate(
            (
                self.sensor.reshape(-1),
                self.collision,
                self.velocity,
                [self.angle, self.energy],
            )
        )

    @property
    def n_collided_foods(self) -> float:
        return self.collision[utils.CollisionType.FOOD.value]

    @property
    def n_collided_agents(self) -> float:
        return self.collision[utils.CollisionType.AGENT.value]


class _CFBodyInfo(NamedTuple):
    position: Vec2d
    velocity: Vec2d


class CFBody(Body[Vec2d]):
    """Body of an agent."""

    def __init__(
        self,
        *,
        body_with_sensors: utils.BodyWithSensors,
        space: pymunk.Space,
        generation: int,
        birthtime: int,
        min_acts: list[float],
        max_acts: list[float],
        max_abs_velocity: float,
        loc: Vec2d,
    ) -> None:
        self._body, self._shape, self._sensors = body_with_sensors
        self._body.position = loc
        space.add(self._body, self._shape, *self._sensors)
        n_sensors = len(self._sensors)
        obs_space = NamedTupleSpace(
            CFObs,
            sensor=BoxSpace(low=0.0, high=1.0, shape=(n_sensors, 3)),
            collision=BoxSpace(low=0.0, high=1.0, shape=(3,)),
            velocity=BoxSpace(low=-max_abs_velocity, high=max_abs_velocity, shape=(2,)),
            angle=BoxSpace(low=0.0, high=2 * np.pi, shape=(1,)),
            energy=BoxSpace(low=0.0, high=50.0, shape=(1,)),
        )
        super().__init__(
            BoxSpace(
                low=np.array(min_acts, dtype=np.float32),
                high=np.array(max_acts, dtype=np.float32),
            ),
            obs_space,
            generation,
            birthtime,
        )

    def info(self) -> Any:
        return _CFBodyInfo(position=self._body.position, velocity=self._body.velocity)

    def _apply_action(self, action: NDArray) -> None:
        fy, impulse_angle = self.act_space.clip(action)
        impulse = Vec2d(0, fy).rotated(impulse_angle)
        self._body.apply_impulse_at_local_point(impulse)

    def _remove(self, space: pymunk.Space) -> None:
        space.remove(self._body, self._shape, *self._sensors)

    def location(self) -> pymunk.vec2d.Vec2d:
        return self._body.position


class AngleCtrlCFBody(CFBody):
    """Agent body that has a different action space (forward force and angle)."""

    _TWO_PI = np.pi * 2

    def _apply_action(self, action: NDArray) -> None:
        action = self.act_space.clip(action)
        if len(action) == 2:
            fy, angle = action
            impulse_angle = 0.0
        else:
            fy, impulse_angle, angle = action
        self._body.angle = (self._body.angle + angle) % self._TWO_PI
        impulse = Vec2d(0, fy).rotated(impulse_angle)
        self._body.apply_impulse_at_local_point(impulse)


def _range(segment: tuple[float, float]) -> float:
    return segment[1] - segment[0]


def _default_energy_function(_body: CFBody) -> float:
    return 0.0


def _get_num_or_loc_fn(
    arg: str | tuple | FN,
    enum_type: Callable[..., Callable[..., FN]],
    default_args: dict[str, tuple[Any, ...]],
) -> FN:
    if isinstance(arg, str):
        return enum_type(arg)(*default_args[arg])
    elif isinstance(arg, tuple) or isinstance(arg, list):
        name, *args = arg
        return enum_type(name)(*args)
    else:
        return arg


class CircleForaging(Env[NDArray, Vec2d, CFObs]):
    _AGENT_COLOR = Color(2, 204, 254)
    _FOOD_COLOR = Color(254, 2, 162)
    _WALL_RADIUS = 0.5

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
        wall_friction: float = 0.05,
        max_abs_impulse: float = 0.2,
        max_abs_angle: float | None = None,
        max_abs_velocity: float = 1.0,
        dt: float = 0.05,
        encount_threshold: int = 2,
        n_physics_steps: int = 5,
        max_place_attempts: int = 10,
        body_elasticity: float = 0.4,
        oned_impulse: bool = False,
        nofriction: bool = False,
        energy_fn: Callable[[CFBody], float] = _default_energy_function,
        seed: int | None = None,
    ) -> None:
        # Just copy some invariable configs
        self._dt = dt
        self._n_physics_steps = n_physics_steps
        self._agent_radius = agent_radius
        self._food_radius = food_radius
        self._n_initial_bodies = n_initial_bodies
        self._max_place_attempts = max_place_attempts
        self._encount_threshold = min(encount_threshold, n_physics_steps)
        self._n_sensors = n_agent_sensors
        self._sensor_length = sensor_length
        self._max_abs_impulse = max_abs_impulse
        self._max_abs_angle = max_abs_angle
        self._max_abs_velocity = max_abs_velocity
        self._oned_impulse = oned_impulse
        self._food_initial_force = food_initial_force
        self._energy_fn = energy_fn

        if env_shape == "square":
            self._coordinate = SquareCoordinate(xlim, ylim, self._WALL_RADIUS)
        elif env_shape == "circle":
            self._coordinate = CircleCoordinate((env_radius, env_radius), env_radius)
        else:
            raise ValueError(f"Unsupported env_shape {env_shape}")

        # nofriction overrides friction values
        if nofriction:
            agent_friction = 0.0
            food_friction = 0.0
            wall_friction = 0.0

        # Save pymunk params in closures
        self._make_pymunk_body = partial(
            utils.circle_body_with_sensors,
            radius=agent_radius,
            n_sensors=n_agent_sensors,
            sensor_length=sensor_length,
            mass=agent_mass,
            friction=agent_friction,
            sensor_range=sensor_range,
            elasticity=body_elasticity,
        )
        self._make_pymunk_food = partial(
            utils.circle_body,
            radius=food_radius,
            collision_type=utils.CollisionType.FOOD,
            mass=food_mass,
            friction=food_friction,
            body_type=pymunk.Body.STATIC,
        )

        # Customizable functions
        self._food_num_fn = self._make_food_num_fn(food_num_fn)
        self._xlim, self._ylim = self._coordinate.bbox()
        self._x_range, self._y_range = _range(xlim), _range(ylim)
        self._food_loc_fn = self._make_food_loc_fn(food_loc_fn)
        self._body_loc_fn = self._make_body_loc_fn(body_loc_fn)
        # Variables
        self._sim_steps = 0
        self._n_foods = 0
        self._space = pymunk.Space()
        # Setup physical objects
        if isinstance(self._coordinate, SquareCoordinate):
            utils.add_static_square(
                self._space,
                *xlim,
                *ylim,
                friction=wall_friction,
                radius=self._WALL_RADIUS,
                rounded_offset=np.floor(food_radius * 2 / (np.sqrt(2) - 1.0)),
            )
        elif isinstance(self._coordinate, CircleCoordinate):
            utils.add_static_approximated_circle(
                self._space,
                self._coordinate.center,
                self._coordinate.radius,
                friction=wall_friction,
            )

        # Set obstacles
        if obstacles is not None:
            for obstacle in obstacles:
                utils.add_static_line(
                    self._space,
                    obstacle[:2],
                    obstacle[2:],
                    friction=wall_friction,
                    radius=self._WALL_RADIUS,
                )
        self._bodies = []
        self._body_indices = {}
        self._foods: dict[pymunk.Body, pymunk.Shape] = {}
        self._encounted_bodies = set()
        self._generator = Generator(PCG64(seed=seed))
        # Shape filter
        self._all_shape = pymunk.ShapeFilter()
        # Place bodies and foods
        self._initialize_bodies_and_foods()
        # Setup all collision handlers
        self._food_handler = utils.FoodHandler(self._body_indices)
        self._mating_handler = utils.MatingHandler(self._body_indices)
        self._static_handler = utils.StaticHandler(self._body_indices)

        utils.add_pre_handler(
            self._space,
            utils.CollisionType.AGENT,
            utils.CollisionType.FOOD,
            self._food_handler,
        )

        utils.add_pre_handler(
            self._space,
            utils.CollisionType.AGENT,
            utils.CollisionType.AGENT,
            self._mating_handler,
        )

        utils.add_pre_handler(
            self._space,
            utils.CollisionType.AGENT,
            utils.CollisionType.STATIC,
            self._static_handler,
        )

    @staticmethod
    def _make_food_num_fn(food_num_fn: str | tuple | ReprNumFn) -> ReprNumFn:
        return _get_num_or_loc_fn(
            food_num_fn,
            ReprNum,
            {"constant": (10,), "logistic": (8, 1.2, 12)},
        )

    def _make_food_loc_fn(self, food_loc_fn: str | tuple | ReprLocFn) -> ReprLocFn:
        return _get_num_or_loc_fn(
            food_loc_fn,
            ReprLoc,
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
                "uniform": (self._coordinate,),
            },
        )

    def _make_body_loc_fn(self, init_loc_fn: str | tuple | InitLocFn) -> InitLocFn:
        return _get_num_or_loc_fn(
            init_loc_fn,
            InitLoc,
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

    def set_body_loc_fn(self, body_loc_fn: str | tuple | InitLocFn) -> None:
        self._body_loc_fn = self._make_body_loc_fn(body_loc_fn)

    def set_energy_fn(self, energy_fn: Callable[[CFBody], float]) -> None:
        self._energy_fn = energy_fn

    def get_space(self) -> pymunk.Space:
        return self._space

    def get_coordinate(self) -> Coordinate:
        return self._coordinate

    def bodies(self) -> list[CFBody]:
        """Return the list of all bodies"""
        return self._bodies

    def step(self, actions: dict[CFBody, NDArray]) -> list[Encount]:
        self._before_step()
        # Add force
        for body, action in actions.items():
            body._apply_action(action)
        # Step the simulation
        for _ in range(self._n_physics_steps):
            self._space.step(dt=self._dt)
        # Remove foods
        n_eaten_foods = len(self._food_handler.eaten_bodies)
        if n_eaten_foods > 0:
            logger.debug(f"{n_eaten_foods} foods are eaten")
            for food_body in self._food_handler.eaten_bodies:
                food_shape = self._foods.pop(food_body)
                self._space.remove(food_body, food_shape)
        # Generate new foods
        locations = [body.position for body in self._foods.keys()]
        n_new_foods = self._food_num_fn(len(locations))
        if n_new_foods > 0:
            n_created = self._place_n_foods(n_new_foods, locations)
            if n_created > 0:
                logger.debug(f"{n_created} foods are created")
        # Increment the step
        self._sim_steps += 1
        return self._all_encounts()

    def observe(self, body: CFBody) -> CFObs:
        """
        Observe the environment.
        More specifically, an observation of each agent consists of:
        - Sensor observation for agent/food/static object
        - Collision to agent/food/static object
        - Velocity of the body
        """
        sensor_data = self._accumulate_sensor_data(body)
        collision_data = np.zeros(3, dtype=np.float32)
        collision_data[0] = body.index in self._encounted_bodies
        collision_data[1] = self._food_handler.n_ate_foods[body.index]
        collision_data[2] = body.index in self._static_handler.collided_bodies

        return CFObs(
            sensor=sensor_data,
            collision=collision_data,
            velocity=body._body.velocity,
            angle=body._body.angle % (2.0 * np.pi),
            energy=self._energy_fn(body),
        )

    def reset(self, seed: int | NDArray | None = None) -> None:
        # Reset indices
        self._sim_steps = 0
        # Remove agents
        for body in self._bodies:
            body._remove(self._space)
        self._bodies.clear()
        # Remove foods
        for body, shape in self._foods.items():
            self._space.remove(body, shape)
        self._foods.clear()
        self._generator = Generator(PCG64(seed=seed))
        self._initialize_bodies_and_foods()

    def locate_body(self, location: Vec2d, generation: int) -> CFBody | None:
        if self._can_place(location, self._agent_radius):
            body = self._make_body(generation=generation, loc=location)
            self._bodies.append(body)
            return body
        else:
            logger.warning(f"Failed to place the body at {location}")
            return None

    def remove_body(self, body: CFBody) -> bool:
        if body._body in self._body_indices:
            body._remove(self._space)
            self._bodies.remove(body)
            del self._body_indices[body._body]
            return True
        else:
            return False

    def is_extinct(self) -> bool:
        return len(self._bodies) == 0

    def visualizer(
        self,
        mode: str,
        figsize: tuple[float, float] | None = None,
        mgl_backend: str = "pyglet",
        **kwargs,
    ) -> Visualizer:
        mode = mode.lower()
        xlim, ylim = self._coordinate.bbox()
        if mode == "pygame":
            from emevo.environments.pymunk_envs import pygame_vis

            return pygame_vis.PygameVisualizer(
                x_range=_range(xlim),
                y_range=_range(ylim),
                figsize=figsize,
                **kwargs,
            )
        elif mode == "moderngl":
            from emevo.environments.pymunk_envs import moderngl_vis

            return moderngl_vis.MglVisualizer(
                x_range=_range(xlim),
                y_range=_range(ylim),
                env=self,
                figsize=figsize,
                backend=mgl_backend,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _accumulate_sensor_data(self, body: CFBody) -> NDArray:
        sensor_data = np.zeros((3, self._n_sensors), dtype=np.float32)
        for i, sensor in enumerate(body._sensors):
            query_result = utils.sensor_query(self._space, body._body, sensor)
            if query_result is not None:
                categ, dist = query_result
                assert categ in [
                    utils.CollisionType.AGENT,
                    utils.CollisionType.FOOD,
                    utils.CollisionType.STATIC,
                ]
                sensor_data[categ.value][i] = 1.0 - dist
        return sensor_data

    def _all_encounts(self) -> list[Encount]:
        all_encounts = []
        for id_a, id_b in self._mating_handler.filter_pairs(self._encount_threshold):
            self._encounted_bodies.add(id_a)
            self._encounted_bodies.add(id_b)
            body_a = self._find_body_by_id(id_a)
            body_b = self._find_body_by_id(id_b)
            all_encounts.append(Encount(body_a, body_b))
        return all_encounts

    def _before_step(self) -> None:
        """Clear all collision handlers before step is called"""
        self._food_handler.clear()
        self._mating_handler.clear()
        self._static_handler.clear()
        self._encounted_bodies.clear()

    def _can_place(self, point: Vec2d, radius: float) -> bool:
        if not self._coordinate.contains_circle(point, radius):
            return False
        nearest = self._space.point_query_nearest(point, radius, self._all_shape)
        return nearest is None

    def _find_body_by_id(self, index: int) -> CFBody:
        for body in self._bodies:
            if body.index == index:
                return body
        raise ValueError(f"Invalid agent index: {index}")

    def _initialize_bodies_and_foods(self) -> None:
        assert len(self._bodies) == 0 and len(self._foods) == 0

        for _ in range(self._n_initial_bodies):
            point = self._try_placing_agent()
            if point is None:
                logger.warning("Failed to place a body")
            else:
                body = self._make_body(generation=0, loc=Vec2d(*point))
                self._bodies.append(body)

        self._place_n_foods(self._food_num_fn.initial)

    @cached_property
    def _min_max_abs_acts(self) -> tuple[list[float], list[float]]:
        if self._max_abs_angle is None or self._max_abs_angle == 0.0:
            return [0.0, -np.pi], [self._max_abs_impulse, np.pi]
        else:
            if self._oned_impulse:
                return [-self._max_abs_impulse, -self._max_abs_angle], [
                    self._max_abs_impulse,
                    self._max_abs_angle,
                ]
            else:
                return [0.0, -np.pi, -self._max_abs_angle], [
                    self._max_abs_impulse,
                    np.pi,
                    self._max_abs_angle,
                ]

    @cached_property
    def _limit_velocity(
        self,
    ) -> Callable[[pymunk.Body, tuple[float, float], float, float], None]:
        return utils.limit_velocity(self._max_abs_velocity)

    def _make_body(self, generation: int, loc: Vec2d) -> CFBody:
        body_with_sensors = self._make_pymunk_body()
        body_with_sensors.shape.color = self._AGENT_COLOR
        body_with_sensors.body.velocify_func = self._limit_velocity
        if self._max_abs_angle is None or self._max_abs_angle == 0.0:
            cls = CFBody
        else:
            cls = AngleCtrlCFBody
        min_acts, max_acts = self._min_max_abs_acts
        fgbody = cls(
            body_with_sensors=body_with_sensors,
            space=self._space,
            generation=generation,
            birthtime=self._sim_steps,
            min_acts=min_acts,
            max_acts=max_acts,
            max_abs_velocity=self._max_abs_velocity,
            loc=loc,
        )
        self._body_indices[body_with_sensors.body] = fgbody.index
        return fgbody

    def _make_food(self, loc: Vec2d) -> tuple[pymunk.Body, pymunk.Shape]:
        body, shape = self._make_pymunk_food()
        shape.color = self._FOOD_COLOR
        if any(map(lambda value: value != 0.0, self._food_initial_force)):
            mean, stddev = self._food_initial_force
            force = self._generator.normal(loc=mean, scale=stddev, size=(2,))
            body.apply_force_at_local_point(Vec2d(*force))
        body.position = loc
        self._space.add(body, shape)
        return body, shape

    def _place_n_foods(
        self,
        n_foods: int,
        food_locations: list[pymunk.Vec2d] | None = None,
    ) -> int:
        if food_locations is None:
            food_locations = []
        success = 0
        for _ in range(n_foods):
            point = self._try_placing_food(food_locations)
            if point is None:
                logger.warning("Failed to place a food")
            else:
                loc = Vec2d(*point)
                food_locations.append(loc)
                food_body, food_shape = self._make_food(loc=loc)
                self._foods[food_body] = food_shape
                success += 1
        return success

    def _try_placing_agent(self) -> NDArray | None:
        for _ in range(self._max_place_attempts):
            sampled = self._body_loc_fn(self._generator)
            if self._can_place(Vec2d(*sampled), self._agent_radius):
                return sampled
        return None

    def _try_placing_food(self, locations: list[Vec2d]) -> NDArray | None:
        for _ in range(self._max_place_attempts):
            sampled = self._food_loc_fn(self._generator, locations)
            if self._can_place(Vec2d(*sampled), self._food_radius):
                return sampled
        return None
