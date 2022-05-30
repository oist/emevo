import abc
import dataclasses
import enum
from collections import defaultdict
from typing import Any, Callable, ClassVar, Dict, Iterable, List, NamedTuple, Set, Tuple
from uuid import UUID

import numpy as np
import pymunk
from pymunk.shapes import Circle


class CollisionType(enum.IntEnum):
    AGENT = 1
    SENSOR = 2
    FOOD = 3
    POISON = 4
    STATIC = 5


def _select(
    shapes: Tuple[pymunk.Shape, pymunk.Shape],
    target_type: CollisionType,
) -> pymunk.Shape:
    for shape in shapes:
        if shape.collision_type == target_type.value:
            return shape
    raise RuntimeError(
        f"Specified collision type {target_type} is not found in {shapes}"
    )


def _clipped_min(value1: float, value2: float) -> float:
    return max(0.0, min(value1, value2))


def add_presolve_handler(
    space: pymunk.Space,
    type_a: CollisionType,
    type_b: CollisionType,
    callback: Callable[[pymunk.arbiter.Arbiter, pymunk.Space, Any], bool],
) -> None:
    collision_handler = space.add_collision_handler(type_a, type_b)
    collision_handler.pre_solve = callback


@dataclasses.dataclass
class FoodHandler:
    """
    Handle collisions between agent and food.
    """

    body_uuids: Dict[pymunk.Body, UUID]
    eaten_bodies: Set[pymunk.Body] = dataclasses.field(default_factory=set)
    n_eaten_foods: Dict[UUID, int] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: 0)
    )

    def __call__(
        self,
        arbiter: pymunk.arbiter.Arbiter,
        _space: pymunk.Space,
        _info: Any,
    ) -> bool:
        """
        Implementation of collision handling callback passed to pymunk.
        Store eaten foods and the number of food an agent ate.
        Return False for already eaten foods.
        """
        a, b = arbiter.shapes
        if a.collision_type == CollisionType.FOOD.value:
            food, agent = a.body, b.body
        else:
            food, agent = b.body, a.body
        if food in self.eaten_bodies:
            return False
        else:
            self.eaten_bodies.add(food)
            uuid = self.body_uuids[agent]
            self.n_eaten_foods[uuid] += 1
            return True

    def clear(self) -> None:
        self.eaten_bodies.clear()
        for uuid in self.n_eaten_foods.keys():
            self.n_eaten_foods[uuid] = 0


@dataclasses.dataclass
class MatingHandler:
    """
    Handle collisions between agents.
    """

    body_uuids: Dict[pymunk.Body, UUID]
    collided_steps: Dict[Tuple[UUID, UUID], int] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: 0)
    )

    def __call__(
        self,
        arbiter: pymunk.arbiter.Arbiter,
        _space: pymunk.Space,
        _info: Any,
    ) -> bool:
        """
        Store collided bodies and the number of collisions per each pair.
        Always return True.
        """
        a, b = map(lambda shape: self.body_uuids[shape.body], arbiter.shapes)
        key = min(a, b), max(a, b)
        self.collided_steps[key] += 1
        return True

    def clear(self) -> None:
        for key in self.collided_steps.keys():
            self.collided_steps[key] = 0

    def filter_pairs(self, threshold: int) -> Iterable[Tuple[UUID, UUID]]:
        """Iterate pairs that collided more than threshold"""
        for pair, n_collided in self.collided_steps.items():
            if threshold <= n_collided:
                yield pair


@dataclasses.dataclass
class StaticHandler:
    """Handle collisions between agents and static objects."""

    body_uuids: Dict[pymunk.Body, UUID]
    collided_bodies: Set[UUID] = dataclasses.field(default_factory=set)

    def __call__(
        self,
        arbiter: pymunk.arbiter.Arbiter,
        _space: pymunk.Space,
        _info: Any,
    ) -> bool:
        """Store collided bodies. Always return True."""
        shape = _select(arbiter.shapes, CollisionType.AGENT)
        self.collided_bodies.add(self.body_uuids[shape.body])
        return True

    def clear(self) -> None:
        self.collided_bodies.clear()


@dataclasses.dataclass
class SensorHandler:
    """
    Handle collisions between sensor and something.
    Here we store only the distance to the object. I.e., we don't distinguish objects.
    """

    # Here distances are reset per each (environment) step.
    # So the use of Shape as key is fine.
    distances: Dict[pymunk.Shape, float] = dataclasses.field(default_factory=dict)
    accumulator: Callable[[float, float], float] = _clipped_min

    def __call__(
        self,
        arbiter: pymunk.arbiter.Arbiter,
        _space: pymunk.Space,
        _info: Any,
    ) -> bool:
        """
        Store accumulated distance for each sensor.
        Always return False.
        """
        assert len(arbiter.contact_point_set.points) == 1
        contact_point = arbiter.contact_point_set.points[0]
        sensor = _select(arbiter.shapes, CollisionType.SENSOR)
        if sensor in self.distances:
            old_dist = self.distances[sensor]
            self.distances[sensor] = self.accumulator(old_dist, contact_point.distance)
        else:
            self.distances[sensor] = contact_point.distance
        return False

    def clear(self) -> None:
        self.distances.clear()


class BodyWithSensors(NamedTuple):
    """Pymunk body with touch sensors."""

    body: pymunk.Body
    shape: pymunk.Shape
    sensors: List[pymunk.Segment]


def limit_velocity(
    max_velocity: float,
) -> Callable[[pymunk.Body, Tuple[float, float], float, float], None]:
    def velocity_callback(
        body: pymunk.Body,
        gravity: Tuple[float, float],
        damping: float,
        dt: float,
    ) -> None:
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        current_velocity = body.velocity.length
        if current_velocity > max_velocity:
            scale = max_velocity / current_velocity
            body.velocity = body.velocity * scale

    return velocity_callback


def circle_body(
    radius: float,
    collision_type: CollisionType,
    mass: float = 1.0,
    friction: float = 0.6,
) -> Tuple[pymunk.Body, Circle]:
    body = pymunk.Body()
    circle = pymunk.Circle(body, radius)
    circle.mass = mass
    circle.friction = friction
    circle.collision_type = collision_type
    return body, circle


def circle_body_with_sensors(
    radius: float,
    n_sensors: int,
    sensor_length: float,
    mass: float = 1.0,
    sensor_width: float = 1.0,
    friction: float = 0.6,
) -> BodyWithSensors:
    body, circle = circle_body(
        radius=radius,
        collision_type=CollisionType.AGENT,
        mass=mass,
        friction=friction,
    )
    sensors = []
    for i in range(n_sensors):
        theta = (2 * np.pi / n_sensors) * i
        x, y = np.cos(theta), np.sin(theta)
        seg = pymunk.Segment(
            body,
            (x * radius, y * radius),
            (x * (radius + sensor_length), (y * (radius + sensor_length))),
            sensor_width,
        )
        seg.sensor = True
        seg.collision_type = CollisionType.SENSOR
        sensors.append(seg)
    return BodyWithSensors(body=body, shape=circle, sensors=sensors)


def add_static_line(
    space: pymunk.Space,
    start: Tuple[float, float],
    end: Tuple[float, float],
    elasticity: float = 0.95,
    friction: float = 0.5,
    radius: float = 1.0,
) -> pymunk.Segment:
    line = pymunk.Segment(space.static_body, start, end, radius)
    line.elasticity = elasticity
    line.friction = friction
    space.add(line)
    return line


def add_static_square(
    space: pymunk.Space,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    **kwargs,
) -> List[pymunk.Segment]:
    p1 = xmin, ymin
    p2 = xmin, ymax
    p3 = xmax, ymax
    p4 = xmax, ymin
    lines = []
    for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
        line = add_static_line(space, start, end, **kwargs)
        lines.append(line)
    return lines
