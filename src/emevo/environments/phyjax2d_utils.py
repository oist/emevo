from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.lax_numpy import TypeVar

from emevo.environments.phyjax2d import (
    Capsule,
    Circle,
    Segment,
    Shape,
    ShapeDict,
    Space,
    StateDict,
    _vmap_dot,
    empty,
)
from emevo.vec2d import Vec2d, Vec2dLike

Self = Any


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
        return jnp.array([jnp.inf]), jnp.array([jnp.inf])
    else:
        return jnp.array(mass), jnp.array(moment)


def _circle_mass(radius: float, density: float) -> tuple[jax.Array, jax.Array]:
    rr = radius**2
    mass = density * jnp.pi * rr
    moment = 0.5 * mass * rr
    return jnp.array([mass]), jnp.array([moment])


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
    return jnp.array([mass]), jnp.array([moment])


S = TypeVar("S", bound=Shape)


def _concat_or(sl: list[S], default_fn: Callable[[], S]) -> S:
    if len(sl) > 0:
        return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *sl)
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


@dataclasses.dataclass
class SpaceBuilder:
    """
    A convenient builder for creating a space.
    Not expected to used with `jax.jit`.
    """

    gravity: Vec2dLike = dataclasses.field(default=(0.0, -9.8))
    circles: list[Circle] = dataclasses.field(default_factory=list)
    static_circles: list[Circle] = dataclasses.field(default_factory=list)
    capsules: list[Capsule] = dataclasses.field(default_factory=list)
    static_capsules: list[Capsule] = dataclasses.field(default_factory=list)
    segments: list[Segment] = dataclasses.field(default_factory=list)
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
            radius=jnp.array([radius]),
            mass=mass,
            moment=moment,
            elasticity=jnp.array([elasticity]),
            friction=jnp.array([friction]),
            rgba=jnp.array(color).reshape(1, 4),
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
            point1=jnp.array(p1).reshape(1, 2),
            point2=jnp.array(p2).reshape(1, 2),
            radius=jnp.array([radius]),
            mass=mass,
            moment=moment,
            elasticity=jnp.array([elasticity]),
            friction=jnp.array([friction]),
            rgba=jnp.array(color).reshape(1, 4),
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
        rgba: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            elasticity=elasticity,
        )
        mass, moment = jnp.array([jnp.inf]), jnp.array([jnp.inf])
        point1 = jnp.array(p1).reshape(1, 2)
        point2 = jnp.array(p2).reshape(1, 2)
        segment = Segment(
            point1=jnp.array(p1).reshape(1, 2),
            point2=jnp.array(p2).reshape(1, 2),
            is_smooth=jnp.array([False]),
            # Fake ghosts
            ghost1=point1,
            ghost2=point2,
            mass=mass,
            moment=moment,
            elasticity=jnp.array([elasticity]),
            friction=jnp.array([friction]),
            rgba=jnp.array(rgba).reshape(1, 4),
        )
        self.segments.append(segment)

    def add_chain_segments(
        self,
        *,
        chain_points: list[tuple[Vec2d, Vec2d]],
        friction: float = 0.8,
        elasticity: float = 0.8,
        rgba: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            elasticity=elasticity,
        )
        mass, moment = jnp.array([jnp.inf]), jnp.array([jnp.inf])
        n_points = len(chain_points)
        for i in range(n_points):
            g1 = chain_points[i - 1][0]
            p1, p2 = chain_points[i]
            g2 = chain_points[(i + 1) % n_points][1]
            segment = Segment(
                point1=jnp.array(p1).reshape(1, 2),
                point2=jnp.array(p2).reshape(1, 2),
                is_smooth=jnp.array([True]),
                # Fake ghosts
                ghost1=jnp.array(g1).reshape(1, 2),
                ghost2=jnp.array(g2).reshape(1, 2),
                mass=mass,
                moment=moment,
                elasticity=jnp.array([elasticity]),
                friction=jnp.array([friction]),
                rgba=jnp.array(rgba).reshape(1, 4),
            )
            self.segments.append(segment)

    def build(self) -> Space:
        shaped = ShapeDict(
            circle=_concat_or(self.circles, empty(Circle)),
            static_circle=_concat_or(self.static_circles, empty(Circle)),
            segment=_concat_or(self.segments, empty(Segment)),
            capsule=_concat_or(self.capsules, empty(Capsule)),
            static_capsule=_concat_or(self.static_capsules, empty(Capsule)),
        )
        dt = self.dt
        linear_damping = jnp.exp(-dt * self.linear_damping).item()
        angular_damping = jnp.exp(-dt * self.angular_damping).item()
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
    center: Vec2dLike,
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


def make_square(
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


def circle_overlap(
    shaped: ShapeDict,
    stated: StateDict,
    xy: jax.Array,
    radius: jax.Array | float,
) -> jax.Array:
    # Circle overlap
    if stated.circle is not None and shaped.circle is not None:
        cpos = stated.circle.p.xy
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        dist = jnp.linalg.norm(cpos - jnp.expand_dims(xy, axis=0), axis=-1)
        penetration = shaped.circle.radius + radius - dist
        has_overlap = jnp.logical_and(stated.circle.is_active, penetration >= 0)
        overlap = jnp.any(has_overlap)
    else:
        overlap = jnp.array(False)

    # Static_circle overlap
    if stated.static_circle is not None and shaped.static_circle is not None:
        cpos = stated.static_circle.p.xy
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        dist = jnp.linalg.norm(cpos - jnp.expand_dims(xy, axis=0), axis=-1)
        penetration = shaped.static_circle.radius + radius - dist
        has_overlap = jnp.logical_and(stated.static_circle.is_active, penetration >= 0)
        overlap = jnp.logical_or(jnp.any(has_overlap), overlap)

    # Circle-segment overlap
    if stated.segment is not None and shaped.segment is not None:
        spos = stated.segment.p
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
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

    return overlap
