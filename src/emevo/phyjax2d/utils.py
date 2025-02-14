"""Utilities to construct physics simulation"""

from __future__ import annotations

import dataclasses
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, NamedTuple, TypeVar
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from emevo.phyjax2d.impl import (
    Capsule,
    Circle,
    Polygon,
    Segment,
    Shape,
    ShapeDict,
    Space,
    State,
    StateDict,
    _vmap_dot,
    empty,
)
from emevo.phyjax2d.vec2d import Vec2d

Self = Any
_N_MAX_POLYGON_VERTICES = 6

_ALL_POLYGON_KEYS = [
    "triangle",
    "static_triangle",
    "quadrangle",
    "static_quadrangle",
    "pentagon",
    "static_pentagon",
]


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: int = 255

    @staticmethod
    def from_float(r: float, g: float, b: float, a: float = 1.0) -> Self:
        return Color(int(r * 255), int(g * 255), int(b * 255), int(a * 255))

    @staticmethod
    def black() -> Self:
        return Color(0, 0, 0, 255)


_BLACK = Color.black()


def _mass_and_moment(
    mass: jax.Array,
    moment: jax.Array,
    is_static: bool = False,
) -> tuple[jax.Array, jax.Array]:
    if is_static:
        return jnp.array(jnp.inf), jnp.array(jnp.inf)
    else:
        return jnp.array(mass), jnp.array(moment)


def _circle_mass(radius: float, density: float) -> tuple[jax.Array, jax.Array]:
    rr = radius**2
    mass = density * jnp.pi * rr
    moment = 0.5 * mass * rr
    return jnp.array(mass), jnp.array(moment)


def _capsule_mass(
    radius: float,
    length: float,
    density: float,
) -> tuple[jax.Array, jax.Array]:
    rr, ll = radius**2, length**2
    mass = density * (jnp.pi * radius + 2.0 * length) * radius
    circle_moment = 0.5 * (rr + ll)
    box_moment = (4 * rr + ll) / 12
    moment = mass * (circle_moment + box_moment)
    return jnp.array(mass), jnp.array(moment)


def _polygon_mass(
    points: list[Vec2d],
    normals: list[Vec2d],
    radius: float,
    density: float,
) -> tuple[jax.Array, jax.Array]:
    vertices = deepcopy(points)
    n = len(vertices)
    # If the polygon is rounded, approximate mass by pushing out vertices
    if radius > 0:
        radius_sqrt2 = radius * (2**0.5)
        for i in range(n):
            j = (i + 1) % n
            n1 = normals[i]
            n2 = normals[j]
            mid = (n1 + n2).normalized()
            vertices[i] += radius_sqrt2 * mid

    r = vertices[0]
    total_area = 0.0
    center = Vec2d(0.0, 0.0)
    moment = 0.0
    for v1, v2 in zip(vertices[1:], vertices[2:]):
        e1 = v1 - r
        e2 = v2 - r
        d = e1.cross(e2)
        area = d * 0.5
        total_area += area
        center += (e1 + e2) * (area / 3)
        intx2 = e1.x**2 + e2.x**2 + e1.x * e2.x
        inty2 = e1.y**2 + e2.y**2 + e1.y * e2.y
        moment += d * (intx2 + inty2) / 12

    mass = density * total_area
    center /= total_area
    center_r = center + r
    moment_shift = center_r.dot(center_r) - center.dot(center)
    moment = (density * moment) + mass * moment_shift
    return jnp.array(mass), jnp.array(moment)


S = TypeVar("S", bound=Shape)


def _stack_or(sl: list[S], default_fn: Callable[[], S]) -> S:
    if len(sl) > 0:
        return jax.tree.map(lambda *args: jnp.stack(args), *sl)
    else:
        return default_fn()


def _check_params_positive(friction: float, **kwargs) -> None:
    if friction > 1.0:
        warnings.warn(
            f"friction larger than 1 can lead instable simulation (value: {friction})",
            stacklevel=2,
        )
    for key, value in kwargs.items():
        assert value > 0.0, f"Invalid value for {key}: {value}"


def _compute_centroid(points: list[Vec2d]) -> Vec2d:
    # Compute centroid
    center = Vec2d(0.0, 0.0)
    total_area = 0.0
    origin = points[0]
    for a, b in zip(points[1:], points[2:]):
        e1 = a - origin
        e2 = b - origin
        area = e1.cross(e2) * 0.5
        center += (e1 + e2) * (area / 3)
        total_area += area
    return (center / total_area) + origin


def _make_listdd() -> dict[int, list]:
    return defaultdict(list)


@dataclasses.dataclass
class SpaceBuilder:
    """
    A convenient builder for creating a space.
    Not expected to used with `jax.jit`.
    """

    gravity: Vec2d | tuple[float, float] = dataclasses.field(default=(0.0, -9.8))
    circles: list[Circle] = dataclasses.field(default_factory=list)
    static_circles: list[Circle] = dataclasses.field(default_factory=list)
    capsules: list[Capsule] = dataclasses.field(default_factory=list)
    static_capsules: list[Capsule] = dataclasses.field(default_factory=list)
    segments: list[Segment] = dataclasses.field(default_factory=list)
    polygons: dict[int, list[Polygon]] = dataclasses.field(default_factory=_make_listdd)
    static_polygons: dict[int, list[Polygon]] = dataclasses.field(
        default_factory=_make_listdd
    )
    dt: float = 0.1
    linear_damping: float = 0.9
    angular_damping: float = 0.9
    bias_factor: float = 0.2
    n_velocity_iter: int = 6
    n_position_iter: int = 2
    linear_slop: float = 0.005
    max_linear_correction: float = 0.2
    allowed_penetration: float = 0.005
    bounce_threshold: float = 1.0
    max_velocity: float | None = None
    max_angular_velocity: float | None = None

    def add_circle(
        self,
        *,
        radius: float,
        density: float = 1.0,
        is_static: bool = False,
        friction: float = 0.8,
        elasticity: float = 0.8,
        color: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            radius=radius,
            density=density,
            elasticity=elasticity,
        )
        mass, moment = _mass_and_moment(*_circle_mass(radius, density), is_static)
        circle = Circle(
            radius=jnp.array(radius),
            mass=mass,
            moment=moment,
            elasticity=jnp.array(elasticity),
            friction=jnp.array(friction),
            rgba=jnp.array(color),
        )
        if is_static:
            self.static_circles.append(circle)
        else:
            self.circles.append(circle)

    def add_capsule(
        self,
        *,
        p1: Vec2d,
        p2: Vec2d,
        radius: float,
        density: float = 1.0,
        is_static: bool = False,
        friction: float = 0.8,
        elasticity: float = 0.8,
        color: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            radius=radius,
            density=density,
            elasticity=elasticity,
        )
        mass, moment = _mass_and_moment(
            *_capsule_mass(radius, (p2 - p1).length, density),
            is_static,
        )
        capsule = Capsule(
            point1=jnp.array(p1),
            point2=jnp.array(p2),
            radius=jnp.array(radius),
            mass=mass,
            moment=moment,
            elasticity=jnp.array(elasticity),
            friction=jnp.array(friction),
            rgba=jnp.array(color),
        )
        if is_static:
            self.static_capsules.append(capsule)
        else:
            self.capsules.append(capsule)

    def add_segment(
        self,
        *,
        p1: Vec2d,
        p2: Vec2d,
        friction: float = 0.8,
        elasticity: float = 0.8,
        color: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            elasticity=elasticity,
        )
        point1 = jnp.array(p1)
        point2 = jnp.array(p2)
        segment = Segment(
            point1=point1,
            point2=point2,
            is_smooth=jnp.array(False),
            # Fake ghosts
            ghost1=point1,
            ghost2=point2,
            mass=jnp.array(jnp.inf),
            moment=jnp.array(jnp.inf),
            elasticity=jnp.array(elasticity),
            friction=jnp.array(friction),
            rgba=jnp.array(color),
        )
        self.segments.append(segment)

    def add_polygon(
        self,
        *,
        points: list[Vec2d],
        radius: float = 0.0,
        friction: float = 0.8,
        elasticity: float = 0.8,
        density: float = 1.0,
        is_static: bool = False,
        color: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            elasticity=elasticity,
        )
        # Need to have at least three points
        n = len(points)
        if n < 3:
            raise ValueError("Polygon needs at least three vertices")
        elif n > _N_MAX_POLYGON_VERTICES:
            raise ValueError(f"Too many vertices {n} for polygon")
        # Check convexity
        cross_list = []
        for a, b, c in zip(points, points[1:], points[2:]):
            a2b = b - a
            b2c = c - b
            cross = a2b.cross(b2c)
            if cross == 0:
                raise ValueError(f"Redundant points in polygon: {a, b, c}")
            cross_list.append(cross)
        signs = np.sign(cross_list)
        if np.all(signs == -1):
            # Flip arrays to make them counter clock wise
            points.reverse()
        elif not np.all(signs == 1):
            raise ValueError("Given polygon is not convex!")
        self._add_polygon_internal(
            points=points,
            centroid=_compute_centroid(points),
            radius=radius,
            friction=friction,
            elasticity=elasticity,
            density=density,
            is_static=is_static,
            rgba=color,
        )

    def add_square(
        self,
        *,
        width: float,
        height: float,
        radius: float = 0.0,
        friction: float = 0.8,
        elasticity: float = 0.8,
        density: float = 1.0,
        is_static: bool = False,
        color: Color = _BLACK,
    ) -> None:
        a = Vec2d(width / 2, height / 2)
        b = Vec2d(-width / 2, height / 2)
        c = Vec2d(-width / 2, -height / 2)
        d = Vec2d(width / 2, -height / 2)
        self._add_polygon_internal(
            points=[a, b, c, d],
            centroid=Vec2d(0.0, 0.0),
            radius=radius,
            friction=friction,
            elasticity=elasticity,
            density=density,
            is_static=is_static,
            rgba=color,
        )

    def _add_polygon_internal(
        self,
        *,
        points: list[Vec2d],
        centroid: Vec2d,
        radius: float = 0.0,
        friction: float = 0.8,
        elasticity: float = 0.8,
        density: float = 1.0,
        is_static: bool = False,
        rgba: Color = _BLACK,
    ) -> None:
        # Compute normal
        normals = []

        for p1, p2 in zip(points, points[1:] + points[:1]):
            edge = p2 - p1
            if edge.dot(edge) < 1e-6:
                raise ValueError(f"Edge is too short in polygon: {p1} and {p2}")
            # Rotate the edge 90 degree right and normalize it
            normals.append(edge.perpendicular_right().normalized())

        mass, moment = _polygon_mass(points, normals, radius, density)
        polygon = Polygon(
            points=jnp.array(points),
            normals=jnp.array(normals),
            centroid=jnp.array(centroid),
            radius=jnp.array(radius),
            mass=mass,
            moment=moment,
            elasticity=jnp.array(elasticity),
            friction=jnp.array(friction),
            rgba=jnp.array(rgba),
        )
        n = len(points)
        if is_static:
            self.static_polygons[n].append(polygon)
        else:
            self.polygons[n].append(polygon)

    def add_chain_segments(
        self,
        *,
        chain_points: list[tuple[Vec2d, Vec2d]],
        friction: float = 0.8,
        elasticity: float = 0.8,
        color: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            elasticity=elasticity,
        )
        n_points = len(chain_points)
        for i in range(n_points):
            g1 = chain_points[i - 1][0]
            p1, p2 = chain_points[i]
            g2 = chain_points[(i + 1) % n_points][1]
            segment = Segment(
                point1=jnp.array(p1),
                point2=jnp.array(p2),
                is_smooth=jnp.array(True),
                # Fake ghosts
                ghost1=jnp.array(g1),
                ghost2=jnp.array(g2),
                mass=jnp.array(jnp.inf),
                moment=jnp.array(jnp.inf),
                elasticity=jnp.array(elasticity),
                friction=jnp.array(friction),
                rgba=jnp.array(color),
            )
            self.segments.append(segment)

    def build(self) -> Space:
        shaped = ShapeDict(
            circle=_stack_or(self.circles, empty(Circle)),
            static_circle=_stack_or(self.static_circles, empty(Circle)),
            segment=_stack_or(self.segments, empty(Segment)),
            capsule=_stack_or(self.capsules, empty(Capsule)),
            static_capsule=_stack_or(self.static_capsules, empty(Capsule)),
            triangle=_stack_or(self.polygons[3], empty(Polygon)),
            static_triangle=_stack_or(self.static_polygons[3], empty(Polygon)),
            quadrangle=_stack_or(self.polygons[4], empty(Polygon)),
            static_quadrangle=_stack_or(self.static_polygons[4], empty(Polygon)),
            pentagon=_stack_or(self.polygons[5], empty(Polygon)),
            static_pentagon=_stack_or(self.static_polygons[5], empty(Polygon)),
            hexagon=_stack_or(self.polygons[6], empty(Polygon)),
            static_hexagon=_stack_or(self.static_polygons[6], empty(Polygon)),
        )
        linear_damping = jnp.exp(-self.dt * self.linear_damping).item()
        angular_damping = jnp.exp(-self.dt * self.angular_damping).item()
        max_velocity = jnp.inf if self.max_velocity is None else self.max_velocity
        max_angular_velocity = (
            jnp.inf if self.max_angular_velocity is None else self.max_angular_velocity
        )
        return Space(
            gravity=jnp.array(self.gravity),
            shaped=shaped,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
            bias_factor=self.bias_factor,
            n_velocity_iter=self.n_velocity_iter,
            n_position_iter=self.n_position_iter,
            linear_slop=self.linear_slop,
            max_linear_correction=self.max_linear_correction,
            allowed_penetration=self.allowed_penetration,
            bounce_threshold=self.bounce_threshold,
            max_velocity=max_velocity,
            max_angular_velocity=max_angular_velocity,
        )


def make_approx_circle(
    center: Vec2d | tuple[float, float],
    radius: float,
    n_lines: int = 32,
) -> list[tuple[Vec2d, Vec2d]]:
    """Make circle. Points are ordered clockwith."""
    unit = np.pi * 2 / n_lines
    lines = []
    t0 = Vec2d(radius, 0.0)
    for i in reversed(range(n_lines)):
        start = center + t0.rotated(unit * (i + 1))
        end = center + t0.rotated(unit * i)
        lines.append((start, end))
    return lines


def make_square_segments(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    rounded_offset: float | None = None,
) -> list[tuple[Vec2d, Vec2d]]:
    """Make square. Points are ordered clockwith."""
    p1 = Vec2d(xmin, ymin)
    p2 = Vec2d(xmin, ymax)
    p3 = Vec2d(xmax, ymax)
    p4 = Vec2d(xmax, ymin)
    lines = []
    if rounded_offset is not None:
        for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            s2end = Vec2d(*end) - Vec2d(*start)
            offset = s2end.normalized() * rounded_offset
            stop = end - offset
            lines.append((start + offset, stop))
            # Center of the rounded corner
            center = stop + offset.rotated(-np.pi / 2)
            for i in reversed(range(4)):
                r_start = center + offset.rotated(np.pi / 8 * (i + 1))
                r_end = center + offset.rotated(np.pi / 8 * i)
                lines.append((r_start, r_end))
    else:
        for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            lines.append((start, end))
    return lines


def _circle_polygon_overlap(
    polygon: Polygon,
    pstate: State,
    xy: jax.Array,
    radius: jax.Array | float,
) -> jax.Array:
    n_batch, n_vertices = polygon.points.shape[:2]
    # Suppose that pstate.p.xy.shape == (N, 2) and xy.shape == (2,)
    cxy = pstate.p.inv_transform(jnp.expand_dims(xy, axis=0))
    p2cxy = jnp.expand_dims(cxy, axis=1) - polygon.points
    separation = _vmap_dot(polygon.normals, p2cxy)  # (N, NP)
    max_sep = jnp.max(separation, axis=1)
    i1 = jnp.argmax(separation, axis=1)
    i2 = (i1 + 1) % n_vertices
    select_all = jnp.arange(n_batch)
    v1 = polygon.points[select_all, i1]
    v2 = polygon.points[select_all, i2]
    u1 = _vmap_dot(cxy - v1, v2 - v1)
    u2 = _vmap_dot(cxy - v2, v1 - v2)
    v = jnp.where(jnp.expand_dims(u1 < 0.0, axis=1), v1, v2)
    dist = jnp.linalg.norm(cxy - v, axis=1)
    c_out = dist < polygon.radius + radius
    c_in = max_sep < polygon.radius + radius
    return jax.lax.select(jnp.logical_or(u1 < 0.0, u2 < 0.0), c_out, c_in)


def circle_overlap(
    shaped: ShapeDict,
    stated: StateDict,
    xy: jax.Array,
    radius: jax.Array | float,
) -> jax.Array:
    # Circle overlap
    overlap = jnp.array(False)
    if not stated.circle.is_empty() and not shaped.circle.is_empty():
        cpos = stated.circle.p.xy
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        dist = jnp.linalg.norm(cpos - jnp.expand_dims(xy, axis=0), axis=-1)
        penetration = shaped.circle.radius + radius - dist
        has_overlap = jnp.logical_and(stated.circle.is_active, penetration >= 0)
        overlap = jnp.any(has_overlap)

    # Static_circle overlap
    if not stated.static_circle.is_empty() and not shaped.static_circle.is_empty():
        cpos = stated.static_circle.p.xy
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        dist = jnp.linalg.norm(cpos - jnp.expand_dims(xy, axis=0), axis=-1)
        penetration = shaped.static_circle.radius + radius - dist
        has_overlap = jnp.logical_and(stated.static_circle.is_active, penetration >= 0)
        overlap = jnp.logical_or(jnp.any(has_overlap), overlap)

    # Circle-segment overlap
    if not stated.segment.is_empty() and not shaped.segment.is_empty():
        spos = stated.segment.p
        # Suppose that spos.shape == (N, 2) and xy.shape == (2,)
        pb = spos.inv_transform(jnp.expand_dims(xy, axis=0))
        p1, p2 = shaped.segment.point1, shaped.segment.point2
        edge = p2 - p1
        s1 = jnp.expand_dims(_vmap_dot(pb - p1, edge), axis=1)
        s2 = jnp.expand_dims(_vmap_dot(p2 - pb, edge), axis=1)
        in_segment = jnp.logical_and(s1 >= 0.0, s2 >= 0.0)
        ee = jnp.sum(jnp.square(edge), axis=-1, keepdims=True)
        pa = jnp.where(in_segment, p1 + edge * s1 / ee, jnp.where(s1 < 0.0, p1, p2))
        dist = jnp.linalg.norm(pb - pa, axis=-1)
        penetration = radius - dist
        has_overlap = jnp.logical_and(stated.segment.is_active, penetration >= 0)
        overlap = jnp.logical_or(jnp.any(has_overlap), overlap)

    # Circle-polygon overlap
    for key in _ALL_POLYGON_KEYS:
        if not stated[key].is_empty() and not shaped[key].is_empty():  # type: ignore
            has_overlap = _circle_polygon_overlap(
                shaped[key],  # type: ignore
                stated[key],  # type: ignore
                xy,
                radius,
            )
            overlap = jnp.logical_or(jnp.any(has_overlap), overlap)

    return overlap
