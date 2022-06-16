from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pymunk
from loguru import logger
from numpy.random import PCG64, Generator
from numpy.typing import NDArray
from pymunk.vec2d import Vec2d

from emevo.body import Body, Encount
from emevo.env import Env, Visualizer
from emevo.environments.pymunk_envs import pymunk_env, pymunk_utils
from emevo.environments.utils.color import Color
from emevo.environments.utils.food_repr import ReprLoc, ReprLocFn, ReprNum, ReprNumFn
from emevo.environments.utils.locating import InitLoc, InitLocFn
from emevo.spaces import BoxSpace, NamedTupleSpace


class FgObs(NamedTuple):
    """Observation of an agent."""

    sensor: NDArray
    collision: NDArray
    velocity: NDArray


class _FgBodyInfo(NamedTuple):
    position: Vec2d
    velocity: Vec2d


class FgBody(Body):
    """Body of an agent."""

    def __init__(
        self,
        *,
        body_with_sensors: pymunk_utils.BodyWithSensors,
        space: pymunk.Space,
        generation: int,
        birthtime: int,
        max_abs_act: float,
        max_abs_velocity: float,
        loc: Vec2d,
    ) -> None:
        self._body, self._shape, self._sensors = body_with_sensors
        self._body.position = loc
        space.add(self._body, self._shape, *self._sensors)
        n_sensors = len(self._sensors)
        act_low = np.ones(2, dtype=np.float32) * -max_abs_act
        act_high = np.ones(2, dtype=np.float32) * max_abs_act
        obs_space = NamedTupleSpace(
            FgObs,
            sensor=BoxSpace(
                low=0.0,
                high=Foraging._SENSOR_MASK_VALUE,
                shape=(n_sensors, 3),
            ),
            collision=BoxSpace(low=0.0, high=1.0, shape=(3,)),
            velocity=BoxSpace(low=-max_abs_velocity, high=max_abs_velocity, shape=(2,)),
        )
        super().__init__(
            BoxSpace(low=act_low, high=act_high),
            obs_space,
            "ForgagingBody",
            generation,
            birthtime,
        )

    def info(self) -> Any:
        return _FgBodyInfo(position=self._body.position, velocity=self._body.velocity)

    def _apply_force(self, force: NDArray) -> None:
        self._body.apply_force_at_local_point(Vec2d(*force))

    def _remove(self, space: pymunk.Space) -> None:
        space.remove(self._body, self._shape, *self._sensors)

    def location(self) -> pymunk.vec2d.Vec2d:
        return self._body.position


def _range(segment: Tuple[float, float]) -> float:
    return segment[1] - segment[0]


_FOOD_NUM_FN_DEFAULT = ReprNum.CONSTANT(10)
_FOOD_LOC_FN_DEFAULT = ReprLoc.GAUSSIAN((150.0, 150.0), (20.0, 20.0))
_BODY_LOC_FN_DEFAULT = InitLoc.UNIFORM((0.0, 0.0), (150, 150))


class Foraging(Env[NDArray, FgBody, Vec2d, FgObs], pymunk_env.PymunkEnv):
    _AGENT_COLOR = Color(2, 204, 254)
    _FOOD_COLOR = Color(254, 2, 162)
    _SENSOR_MASK_VALUE = 2.0
    _WALL_RADIUS = 1.0

    def __init__(
        self,
        n_initial_bodies: int = 6,
        food_num_fn: ReprNumFn = _FOOD_NUM_FN_DEFAULT,
        food_loc_fn: ReprLocFn = _FOOD_LOC_FN_DEFAULT,
        body_loc_fn: InitLocFn = _BODY_LOC_FN_DEFAULT,
        xlim: Tuple[float, float] = (0.0, 200.0),
        ylim: Tuple[float, float] = (0.0, 200.0),
        n_agent_sensors: int = 8,
        sensor_length: float = 10.0,
        sensor_range: Tuple[float, float] = (-180.0, 180.0),
        agent_radius: float = 8.0,
        agent_mass: float = 1.4,
        agent_friction: float = 0.6,
        food_radius: float = 4.0,
        food_mass: float = 0.25,
        food_friction: float = 0.6,
        food_initial_force: Optional[Tuple[float, float]] = None,
        max_abs_force: float = 1.0,
        max_abs_velocity: float = 1.4142135623730951,
        dt: float = 0.05,
        encount_threshold: int = 2,
        n_physics_steps: int = 10,
        max_place_attempts: int = 10,
        seed: Optional[int] = None,
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
        self._max_abs_force = max_abs_force
        self._max_abs_velocity = max_abs_velocity
        self._xlim = xlim
        self._ylim = ylim
        self._food_initial_force = food_initial_force
        # Save pymunk params in closures
        self._make_pymunk_body = partial(
            pymunk_utils.circle_body_with_sensors,
            radius=agent_radius,
            n_sensors=n_agent_sensors,
            sensor_length=sensor_length,
            mass=agent_mass,
            friction=agent_friction,
            sensor_range=sensor_range,
        )
        self._make_pymunk_food = partial(
            pymunk_utils.circle_body,
            radius=food_radius,
            collision_type=pymunk_utils.CollisionType.FOOD,
            mass=food_mass,
            friction=food_friction,
        )
        # Customizable functions
        self._food_num_fn = food_num_fn
        self._food_loc_fn = food_loc_fn
        self._body_loc_fn = body_loc_fn
        # Variables
        self._sim_steps = 0
        self._n_foods = 0
        # Make pymunk world and add bodies
        self._space = pymunk.Space()
        # Setup physical objects
        pymunk_utils.add_static_square(
            self._space,
            *xlim,
            *ylim,
            friction=0.4,
            radius=self._WALL_RADIUS,
        )
        self._bodies = []
        self._body_uuids = {}
        self._foods: Dict[pymunk.Body, pymunk.Shape] = {}
        self._encounted_bodies = set()
        self._generator = Generator(PCG64(seed=seed))
        # Shape filter
        self._all_shape = pymunk.ShapeFilter()
        # Place bodies and foods
        self._initialize_bodies_and_foods()
        # Setup all collision handlers
        self._food_handler = pymunk_utils.FoodHandler(self._body_uuids)
        self._mating_handler = pymunk_utils.MatingHandler(self._body_uuids)
        self._static_handler = pymunk_utils.StaticHandler(self._body_uuids)

        pymunk_utils.add_pre_handler(
            self._space,
            pymunk_utils.CollisionType.AGENT,
            pymunk_utils.CollisionType.FOOD,
            self._food_handler,
        )

        pymunk_utils.add_pre_handler(
            self._space,
            pymunk_utils.CollisionType.AGENT,
            pymunk_utils.CollisionType.AGENT,
            self._mating_handler,
        )

        pymunk_utils.add_pre_handler(
            self._space,
            pymunk_utils.CollisionType.AGENT,
            pymunk_utils.CollisionType.STATIC,
            self._static_handler,
        )

    def get_space(self) -> pymunk.Space:
        return self._space

    def bodies(self) -> List[FgBody]:
        """Return the list of all bodies"""
        return self._bodies

    def step(self, actions: Dict[FgBody, NDArray]) -> List[Encount]:
        self._before_step()
        # Add force
        for body, action in actions.items():
            body._apply_force(action)
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

    def observe(self, body: FgBody) -> FgObs:
        """
        Observe the environment.
        More specifically, an observation of each agent consists of:
        - Sensor observation for agent/food/static object
        - Collision to agent/food/static object
        - Velocity of the body
        """
        sensor_data = self._accumulate_sensor_data(body)
        collision_data = np.zeros(3, dtype=np.float32)
        collision_data[0] = body.uuid in self._encounted_bodies
        collision_data[1] = min(self._food_handler.n_eaten_foods[body.uuid], 1)
        collision_data[2] = body.uuid in self._static_handler.collided_bodies

        return FgObs(
            sensor=sensor_data,
            collision=collision_data,
            velocity=np.array(body._body.velocity),
        )

    def reset(self, seed: Optional[Union[NDArray, int]] = None) -> None:
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

    def born(self, location: Vec2d, generation: int) -> Optional[FgBody]:
        if self._can_place(location, self._agent_radius):
            body = self._make_body(generation=generation, loc=location)
            self._bodies.append(body)
            return body
        else:
            logger.warning(f"Failed to place the body at {location}")
            return None

    def dead(self, body: FgBody) -> None:
        body._remove(self._space)
        self._bodies.remove(body)
        del self._body_uuids[body._body]

    def is_extinct(self) -> bool:
        return len(self._bodies) == 0

    def visualizer(
        self,
        mode: str,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Visualizer:
        mode = mode.lower()
        if mode == "mpl":
            from emevo.environments.pymunk_envs import mpl_vis

            return mpl_vis.MplVisualizer(
                xlim=self._xlim,
                ylim=self._ylim,
                figsize=figsize,
            )
        elif mode == "pygame":
            from emevo.environments.pymunk_envs import pygame_vis

            return pygame_vis.PygameVisualizer(
                x_range=_range(self._xlim),
                y_range=_range(self._ylim),
                figsize=figsize,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _accumulate_sensor_data(self, body: FgBody) -> NDArray:
        sensor_data = (
            np.ones((3, self._n_sensors), dtype=np.float32) * self._SENSOR_MASK_VALUE
        )
        for i, sensor in enumerate(body._sensors):
            query_result = pymunk_utils.sensor_query(self._space, body._body, sensor)
            if query_result is not None:
                categ, dist = query_result
                assert categ in [
                    pymunk_utils.CollisionType.AGENT,
                    pymunk_utils.CollisionType.FOOD,
                    pymunk_utils.CollisionType.STATIC,
                ]
                sensor_data[categ.value][i] = dist
        return sensor_data

    def _all_encounts(self) -> List[Encount]:
        all_encounts = []
        for uuid_a, uuid_b in self._mating_handler.filter_pairs(
            self._encount_threshold
        ):
            self._encounted_bodies.add(uuid_a)
            self._encounted_bodies.add(uuid_b)
            body_a = self._find_body_by_uuid(uuid_a)
            body_b = self._find_body_by_uuid(uuid_b)
            all_encounts.append(Encount(body_a, body_b))
        return all_encounts

    def _before_step(self) -> None:
        """Clear all collision handlers before step is called"""
        self._food_handler.clear()
        self._mating_handler.clear()
        self._static_handler.clear()
        self._encounted_bodies.clear()

    def _can_place(self, point: Vec2d, radius: float) -> bool:
        nearest = self._space.point_query_nearest(
            point,
            radius,
            self._all_shape,
        )
        return nearest is None

    def _find_body_by_uuid(self, uuid: UUID) -> FgBody:
        for body in self._bodies:
            if body.uuid == uuid:
                return body
        raise ValueError(f"Invalid agent uuid: {uuid}")

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

    def _make_body(self, generation: int, loc: Vec2d) -> FgBody:
        body_with_sensors = self._make_pymunk_body()
        body_with_sensors.shape.color = self._AGENT_COLOR
        body_with_sensors.body.velocify_func = pymunk_utils.limit_velocity(
            self._max_abs_velocity
        )
        fgbody = FgBody(
            body_with_sensors=body_with_sensors,
            space=self._space,
            generation=generation,
            birthtime=self._sim_steps,
            max_abs_act=self._max_abs_force,
            max_abs_velocity=self._max_abs_velocity,
            loc=loc,
        )
        self._body_uuids[body_with_sensors.body] = fgbody.uuid
        return fgbody

    def _make_food(self, loc: Vec2d) -> Tuple[pymunk.Body, pymunk.Shape]:
        body, shape = self._make_pymunk_food()
        shape.color = self._FOOD_COLOR
        if self._food_initial_force is not None:
            mean, stddev = self._food_initial_force
            force = self._generator.normal(loc=mean, scale=stddev, size=(2,))
            body.apply_force_at_local_point(Vec2d(*force))
        body.position = loc
        self._space.add(body, shape)
        return body, shape

    def _place_n_foods(
        self,
        n_foods: int,
        food_locations: Optional[List[pymunk.Vec2d]] = None,
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
                food_body, food_shape = self._make_food(loc=Vec2d(*point))
                self._foods[food_body] = food_shape
                success += 1
        return success

    def _try_placing_agent(self) -> Optional[NDArray]:
        for _ in range(self._max_place_attempts):
            sampled = self._body_loc_fn(self._generator)
            if self._can_place(Vec2d(*sampled), self._agent_radius):
                return sampled
        return None

    def _try_placing_food(self, locations: List[Vec2d]) -> Optional[NDArray]:
        for _ in range(self._max_place_attempts):
            sampled = self._food_loc_fn(self._generator, locations)
            if self._can_place(Vec2d(*sampled), self._food_radius):
                return sampled
        return None
