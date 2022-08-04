from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict
from typing import Any, Callable, Iterable, NamedTuple
from uuid import UUID

import numpy as np
import pymunk
from pymunk.shapes import Circle

SENSOR_OFFSET: float = 1e-6


class CollisionType(enum.IntEnum):
    AGENT = 0
    FOOD = 1
    STATIC = 2
    POISON = 3
    SENSOR = 4

    def categ_filter(self) -> pymunk.ShapeFilter:
        return pymunk.ShapeFilter(categories=1 << self.value)


def _select(
    shapes: tuple[pymunk.Shape, pymunk.Shape],
    target_type: CollisionType,
) -> pymunk.Shape:
    for shape in shapes:
        if shape.collision_type == target_type.value:
            return shape
    raise RuntimeError(f"Collision type {target_type} is not found in {shapes}")


def add_pre_handler(
    space: pymunk.Space,
    type_a: CollisionType,
    type_b: CollisionType,
    callback: Callable[[pymunk.arbiter.Arbiter, pymunk.Space, Any], bool],
) -> None:
    """Add pre_solve handler to the space."""
    collision_handler = space.add_collision_handler(type_a.value, type_b.value)
    collision_handler.pre_solve = callback


@dataclasses.dataclass
class FoodHandler:
    """
    Handle collisions between agent and food.
    """

    body_uuids: dict[pymunk.Body, UUID]
    eaten_bodies: set[pymunk.Body] = dataclasses.field(default_factory=set)
    n_eaten_foods: dict[UUID, int] = dataclasses.field(
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

    body_uuids: dict[pymunk.Body, UUID]
    collided_steps: dict[tuple[UUID, UUID], int] = dataclasses.field(
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

    def filter_pairs(self, threshold: int) -> Iterable[tuple[UUID, UUID]]:
        """Iterate pairs that collided more than threshold"""
        for pair, n_collided in self.collided_steps.items():
            if threshold <= n_collided:
                yield pair


@dataclasses.dataclass
class StaticHandler:
    """Handle collisions between agents and static objects."""

    body_uuids: dict[pymunk.Body, UUID]
    collided_bodies: set[UUID] = dataclasses.field(default_factory=set)

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


def sensor_query(
    space: pymunk.Space,
    body: pymunk.Body,
    segment: pymunk.Segment,
    mask: int = pymunk.ShapeFilter.ALL_MASKS() ^ (1 << CollisionType.SENSOR.value),
) -> tuple[CollisionType, float] | None:
    """Get the nearest object aligned with given segment"""
    start = body.position + segment.a
    end = body.position + segment.b
    shape_filter = pymunk.ShapeFilter(mask=mask)
    query_result = space.segment_query_first(start, end, 0.0, shape_filter)
    if query_result is None or query_result.shape is None:
        return None
    else:
        collision_type = CollisionType(query_result.shape.collision_type)
        return collision_type, query_result.alpha


class BodyWithSensors(NamedTuple):
    """Pymunk body with touch sensors."""

    body: pymunk.Body
    shape: pymunk.Shape
    sensors: list[pymunk.Segment]


def limit_velocity(
    max_velocity: float,
) -> Callable[[pymunk.Body, tuple[float, float], float, float], None]:
    def velocity_callback(
        body: pymunk.Body,
        gravity: tuple[float, float],
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
) -> tuple[pymunk.Body, Circle]:
    body = pymunk.Body()
    circle = pymunk.Circle(body, radius)
    circle.mass = mass
    circle.friction = friction
    circle.collision_type = collision_type
    circle.filter = collision_type.categ_filter()
    return body, circle


def circle_body_with_sensors(
    radius: float,
    n_sensors: int,
    sensor_length: float,
    mass: float = 1.0,
    friction: float = 0.6,
    sensor_range: tuple[float, float] = (-180, 180),
) -> BodyWithSensors:
    body, circle = circle_body(
        radius=radius,
        collision_type=CollisionType.AGENT,
        mass=mass,
        friction=friction,
    )
    sensors = []
    sensor_rad = np.deg2rad(sensor_range)
    for theta in np.linspace(sensor_rad[0], sensor_rad[1], n_sensors + 1)[:-1]:
        x, y = np.cos(theta), np.sin(theta)
        seg = pymunk.Segment(
            body,
            (x * (radius + SENSOR_OFFSET), y * (radius + SENSOR_OFFSET)),
            (x * (radius + sensor_length), (y * (radius + sensor_length))),
            0.5,  # This is not used
        )
        seg.sensor = True
        seg.collision_type = CollisionType.SENSOR
        seg.filter = pymunk.ShapeFilter(categories=CollisionType.SENSOR.value, mask=0)
        sensors.append(seg)
    return BodyWithSensors(body=body, shape=circle, sensors=sensors)


def add_static_line(
    space: pymunk.Space,
    start: tuple[float, float],
    end: tuple[float, float],
    elasticity: float = 0.95,
    friction: float = 0.5,
    radius: float = 1.0,
) -> pymunk.Segment:
    line = pymunk.Segment(space.static_body, start, end, radius)
    line.elasticity = elasticity
    line.friction = friction
    line.collision_type = CollisionType.STATIC
    line.filter = CollisionType.STATIC.categ_filter()
    space.add(line)
    return line


def add_static_square(
    space: pymunk.Space,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    **kwargs,
) -> list[pymunk.Segment]:
    p1 = xmin, ymin
    p2 = xmin, ymax
    p3 = xmax, ymax
    p4 = xmax, ymin
    lines = []
    for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
        line = add_static_line(space, start, end, **kwargs)
        lines.append(line)
    return lines
