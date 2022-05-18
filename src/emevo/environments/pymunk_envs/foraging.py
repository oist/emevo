import dataclasses

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import pymunk

from loguru import logger
from numpy.random import PCG64, Generator
from numpy.typing import NDArray
from pymunk.vec2d import Vec2d

from emevo.body import Body, Encount
from emevo.env import Env, Observation
from emevo.environments.pymunk_envs import pymunk_utils
from emevo.environments.utils.food_repr import ReprLoc, ReprLocFn, ReprNum, ReprNumFn
from emevo.environments.utils.locating import InitLoc, InitLocFn
from emevo.types import Info, Location, Shape


class FgBody(Body):
    def __init__(
        self,
        body_with_sensors: pymunk_utils.BodyWithSensors,
        space: pymunk.Space,
        generation: int,
        birthtime: int,
        index: int,
    ) -> None:
        super().__init__("ForgagingBody", generation, birthtime, index)
        self._body, self._shape, self._sensors = body_with_sensors
        space.add(self._body, self._shape, *self._sensors)

    def _apply_force(self, force: NDArray) -> None:
        self._body.apply_force_at_local_point(Vec2d(*force))

    def _remove(self, space: pymunk.Space) -> None:
        space.remove(self._body, self._shape, *self._sensors)

    def act_shape(self) -> Shape:
        return (2,)

    def obs_shape(self) -> Shape:
        return (len(self._sensors),)

    def location(self) -> pymunk.vec2d.Vec2d:
        return self._body.position


class FgFood:
    def __init__(
        self,
        space: pymunk.Space,
        body: pymunk.Body,
        shape: pymunk.Shape,
        loc: Vec2d,
    ) -> None:
        self._body = body
        self._body.position = loc
        self._shape = shape
        space.add(body, shape)

    def _remove(self, space: pymunk.Space) -> None:
        space.remove(self._body, self._shape)


@dataclasses.dataclass
class FgObs(Observation):
    sensor: NDArray

    def as_array(self, source: NDArray) -> NDArray:
        pass


class Foraging(Env[NDArray, FgBody, FgObs]):
    def __init__(
        self,
        n_initial_bodies: int = 6,
        food_num_fn: ReprNumFn = ReprNum.CONSTANT(10),
        food_loc_fn: ReprLocFn = ReprLoc.GAUSSIAN((350.0, 350.0), (10.0, 10.0)),
        body_loc_fn: InitLocFn = InitLoc.UNIFORM((0.0, 0.0), (320, 320)),
        xlim: Tuple[float, float] = (0.0, 400.0),
        ylim: Tuple[float, float] = (0.0, 400.0),
        n_agent_sensors: int = 8,
        sensor_length: float = 6.0,
        agent_radius: float = 4.0,
        agent_mass: float = 2.0,
        food_radius: float = 1.0,
        food_mass: float = 0.5,
        dt: float = 0.05,
        n_physics_steps: int = 10,
        max_place_attempts: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        # Just copy invariable some configs
        self._dt = dt
        self._n_physics_steps = n_physics_steps
        self._agent_radius = agent_radius
        self._food_radius = food_radius
        self._n_initial_bodies = n_initial_bodies
        self._max_place_attempts = max_place_attempts
        # Save pymunk params in closures
        self._make_pymunk_body = partial(
            pymunk_utils.circle_body_with_sensors,
            radius=agent_radius,
            n_sensors=n_agent_sensors,
            sensor_length=sensor_length,
            mass=agent_mass,
            friction=0.6,
        )
        self._make_pymunk_food = partial(
            pymunk_utils.circle_body,
            radius=food_radius,
            collision_type=pymunk_utils.CollisionType.FOOD,
            mass=food_mass,
            friction=0.6,
        )
        # Custimizable functions
        self._food_num_fn = food_num_fn
        self._food_loc_fn = food_loc_fn
        self._body_loc_fn = body_loc_fn
        # Variables
        self._agent_index = 0
        self._sim_steps = 0
        self._n_foods = 0
        # Make pymunk world and add bodies
        self._space = pymunk.Space()
        # Add walls
        pymunk_utils.add_static_square(self._space, *xlim, *ylim, friction=0.4)
        self._bodies = []
        self._foods = []
        self._generator = Generator(PCG64(seed=seed))
        # Shape filter
        self._all_shape = pymunk.ShapeFilter()

    def bodies(self) -> List[FgBody]:
        return self._bodies

    def step(self, actions: Dict[FgBody, NDArray]) -> Tuple[List[Encount], Info]:
        for body, action in actions.items():
            body._apply_force(action)
        for _ in range(self._n_physics_steps):
            self._space.step(dt=self._dt)

    def observe(self, body: Body) -> Tuple[FgObs, Info]:
        pass

    def reset(self, seed: Optional[Union[NDArray, int]] = None) -> None:
        # Reset indices
        self._sim_steps = 0
        # Remove agents
        for body in self._bodies:
            body._remove(self._space)
        self._bodies.clear()
        # Remove foods
        for food in self._foods:
            food._remove(self._space)
        self._foods.clear()
        self._generator = Generator(PCG64(seed=seed))

    def born(self, location: Location) -> Optional[BODY]:
        pass

    def dead(self, body: FgBody) -> None:
        body._remove(self._space)
        self._bodies.remove(body)

    def is_extinct(self) -> bool:
        return len(self._bodies) == 0

    def _make_body(self, generation: int, loc: Vec2d) -> FgBody:
        body = self._make_pymunk_body()
        index = self._agent_index
        self._agent_index += 1
        return FgBody(body, self._space, generation, self._sim_steps, index)

    def _make_food(self, loc: Vec2d) -> FgFood:
        body, shape = self._make_pymunk_food()
        return FgFood(self._space, body, shape, loc)

    def _can_place(
        self, point: Union[Tuple[float, float], NDArray], radius: float
    ) -> bool:
        nearest = self._space.point_query_nearest(
            tuple(point),
            radius,
            self._all_shape,
        )
        return nearest is None

    def _try_placing_agent(self) -> Optional[NDArray]:
        for _ in range(self._max_place_attempts):
            sampled = self._body_loc_fn(self._generator)
            if self._can_place(sampled, self._agent_radius):
                return sampled
        return None

    def _try_placing_food(self, locations: List[Vec2d]) -> Optional[NDArray]:
        for _ in range(self._max_place_attempts):
            sampled = self._food_loc_fn(self._generator, locations)
            if self._can_place(sampled, self._food_radius):
                return sampled
        return None

    def _initialize_bodies_and_foods(self) -> None:
        assert len(self._bodies) == 0 and len(self._foods) == 0

        for _ in range(self._n_initial_bodies):
            point = self._try_placing_agent()
            if point is None:
                logger.warning("Failed to place a body")
            else:
                body = self._make_body(generation=0, loc=Vec2d(*point))
                self._bodies.append(body)

        food_locations = []
        for _ in range(self._food_num_fn.initial):
            point = self._try_placing_food(food_locations)
            if point is None:
                logger.warning("Failed to place a food")
            else:
                loc = Vec2d(*point)
                food_locations.append(loc)
                food = self._make_food(loc=Vec2d(*point))
                self._foods.append(food)
