from __future__ import annotations

import dataclasses
import enum
from collections import defaultdict
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np
import pymunk
from pymunk.body import Vec2d
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

    body_indices: dict[pymunk.Body, int]
    eaten_bodies: set[pymunk.Body] = dataclasses.field(default_factory=set)
    n_ate_foods: dict[int, int] = dataclasses.field(
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
            index = self.body_indices[agent]
            self.n_ate_foods[index] += 1
            return True

    def clear(self) -> None:
        self.eaten_bodies.clear()
        for index in self.n_ate_foods.keys():
            self.n_ate_foods[index] = 0


@dataclasses.dataclass
class MatingHandler:
    """
    Handle collisions between agents.
    """

    body_indices: dict[pymunk.Body, int]
    collided_steps: dict[tuple[int, int], int] = dataclasses.field(
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
        a, b = map(lambda shape: self.body_indices[shape.body], arbiter.shapes)
        key = min(a, b), max(a, b)
        self.collided_steps[key] += 1
        return True

    def clear(self) -> None:
        for key in self.collided_steps.keys():
            self.collided_steps[key] = 0

    def filter_pairs(self, threshold: int) -> Iterable[tuple[int, int]]:
        """Iterate pairs that collided more than threshold"""
        for pair, n_collided in self.collided_steps.items():
            if threshold <= n_collided:
                yield pair


@dataclasses.dataclass
class StaticHandler:
    """Handle collisions between agents and static objects."""

    body_indices: dict[pymunk.Body, int]
    collided_bodies: set[int] = dataclasses.field(default_factory=set)

    def __call__(
        self,
        arbiter: pymunk.arbiter.Arbiter,
        _space: pymunk.Space,
        _info: Any,
    ) -> bool:
        """Store collided bodies. Always return True."""
        shape = _select(arbiter.shapes, CollisionType.AGENT)
        self.collided_bodies.add(self.body_indices[shape.body])
        return True

    def clear(self) -> None:
        self.collided_bodies.clear()


_DEFAULT_MASK = pymunk.ShapeFilter.ALL_MASKS() ^ (1 << CollisionType.SENSOR.value)


def sensor_query(
    space: pymunk.Space,
    body: pymunk.Body,
    segment: pymunk.Segment,
    mask: int = _DEFAULT_MASK,
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
            body.velocity = body.velocity * max_velocity / current_velocity

    return velocity_callback


def circle_body(
    radius: float,
    collision_type: CollisionType,
    mass: float = 1.0,
    friction: float = 0.6,
    elasticity: float = 0.0,
    body_type: int = pymunk.Body.DYNAMIC,
) -> tuple[pymunk.Body, Circle]:
    body = pymunk.Body(body_type=body_type)
    circle = pymunk.Circle(body, radius)
    circle.mass = mass
    circle.friction = friction
    circle.collision_type = collision_type
    circle.filter = collision_type.categ_filter()
    circle.elasticity = elasticity
    return body, circle


def circle_body_with_sensors(
    radius: float,
    n_sensors: int,
    sensor_length: float,
    mass: float = 1.0,
    friction: float = 0.6,
    elasticity: float = 0.0,
    sensor_range: tuple[float, float] = (-180, 180),
) -> BodyWithSensors:
    body, circle = circle_body(
        radius=radius,
        collision_type=CollisionType.AGENT,
        mass=mass,
        friction=friction,
        elasticity=elasticity,
    )
    sensors = []
    sensor_rad = np.deg2rad(sensor_range)
    sensor_in = Vec2d(0.0, radius + SENSOR_OFFSET)
    sensor_out = Vec2d(0.0, radius + sensor_length)
    for theta in np.linspace(sensor_rad[0], sensor_rad[1], n_sensors + 1)[:-1]:
        seg = pymunk.Segment(
            body,
            sensor_in.rotated(theta),
            sensor_out.rotated(theta),
            0.5,
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
    rounded_offset: float | None = None,
    **kwargs,
) -> list[pymunk.Segment]:
    p1 = xmin, ymin
    p2 = xmin, ymax
    p3 = xmax, ymax
    p4 = xmax, ymin
    lines = []
    if rounded_offset is not None:
        for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            s2end = Vec2d(*end) - Vec2d(*start)
            offset = s2end.normalized() * rounded_offset
            stop = end - offset
            line = add_static_line(
                space,
                start + offset,
                stop,
                **kwargs,
            )
            lines.append(line)
            stop2end = end - stop
            center = stop + stop2end.rotated(-np.pi / 2)
            for i in range(4):
                line = add_static_line(
                    space,
                    center + stop2end.rotated(np.pi / 8 * i),
                    center + stop2end.rotated(np.pi / 8 * (i + 1)),
                    **kwargs,
                )
                lines.append(line)
    else:
        for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            line = add_static_line(space, start, end, **kwargs)
            lines.append(line)
    return lines


def add_static_approximated_circle(
    space: pymunk.Space,
    center: tuple[float, float],
    radius: float,
    n_lines: int = 32,
    **kwargs,
) -> list[pymunk.Segment]:
    unit = np.pi * 2 / n_lines
    lines = []
    t0 = Vec2d(radius, 0.0)
    for i in range(n_lines):
        line = add_static_line(
            space,
            center + t0.rotated(unit * i),
            center + t0.rotated(unit * (i + 1)),
            **kwargs,
        )
        lines.append(line)
    return lines
