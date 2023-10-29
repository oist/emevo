import dataclasses
import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from emevo.environments.phyjax2d import (
    Capsule,
    Circle,
    Segment,
    Shape,
    ShapeDict,
    Space,
    State,
    StateDict,
    _length_to_points,
    _vmap_dot,
    circle_raycast,
    segment_raycast,
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
    mass: float = 1.0,
    moment: float = 1.0,
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
        radius: float,
        length: float,
        density: float = 1.0,
        is_static: bool = False,
        friction: float = 0.8,
        elasticity: float = 0.8,
        color: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            radius=radius,
            length=length,
            density=density,
            elasticity=elasticity,
        )
        mass, moment = _mass_and_moment(
            *_capsule_mass(radius, length, density),
            is_static,
        )
        capsule = Capsule(
            length=jnp.array([length]),
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
        length: float,
        friction: float = 0.8,
        elasticity: float = 0.8,
        rgba: Color = _BLACK,
    ) -> None:
        _check_params_positive(
            friction=friction,
            length=length,
            elasticity=elasticity,
        )
        mass, moment = _mass_and_moment(is_static=True)
        segment = Segment(
            length=jnp.array([length]),
            mass=mass,
            moment=moment,
            elasticity=jnp.array([elasticity]),
            friction=jnp.array([friction]),
            rgba=jnp.array(rgba).reshape(1, 4),
        )
        self.segments.append(segment)

    def build(self) -> Space:
        def concat_or(sl: list[Shape]) -> Shape | None:
            if len(sl) > 0:
                return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *sl)
            else:
                return None

        shaped = ShapeDict(
            circle=concat_or(self.circles),
            static_circle=concat_or(self.static_circles),
            segment=concat_or(self.segments),
            capsule=concat_or(self.capsules),
            static_capsule=concat_or(self.static_capsules),
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
    unit = np.pi * 2 / n_lines
    lines = []
    t0 = Vec2d(radius, 0.0)
    for i in range(n_lines):
        start = center + t0.rotated(unit * i)
        end = center + t0.rotated(unit * (i + 1))
        lines.append((start, end))
    return lines


def make_square(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    rounded_offset: float | None = None,
) -> list[tuple[Vec2d, Vec2d]]:
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
            stop2end = end - stop
            center = stop + stop2end.rotated(-np.pi / 2)
            for i in range(4):
                start = center + stop2end.rotated(np.pi / 8 * i)
                end = center + stop2end.rotated(np.pi / 8 * (i + 1))
                lines.append((start, end))
    else:
        for start, end in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            lines.append((start, end))
    return lines


def circle_overwrap(
    shaped: ShapeDict,
    stated: StateDict,
    xy: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    # Circle-circle overwrap

    if stated.circle is not None and shaped.circle is not None:
        cpos = stated.circle.p.xy
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        dist = jnp.linalg.norm(cpos - jnp.expand_dims(xy, axis=0), axis=-1)
        penetration = shaped.circle.radius + radius - dist
        has_overwrap = jnp.logical_and(stated.circle.is_active, penetration >= 0)
        overwrap2cir = jnp.any(has_overwrap)
    else:
        overwrap2cir = jnp.array(False)

    # Circle-static_circle overwrap
    if stated.static_circle is not None and shaped.static_circle is not None:
        cpos = stated.static_circle.p.xy
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        dist = jnp.linalg.norm(cpos - jnp.expand_dims(xy, axis=0), axis=-1)
        penetration = shaped.static_circle.radius + radius - dist
        has_overwrap = jnp.logical_and(stated.static_circle.is_active, penetration >= 0)
        overwrap2cir = jnp.logical_or(jnp.any(has_overwrap), overwrap2cir)

    # Circle-segment overwrap

    if stated.segment is not None and shaped.segment is not None:
        spos = stated.segment.p
        # Suppose that cpos.shape == (N, 2) and xy.shape == (2,)
        pb = spos.inv_transform(jnp.expand_dims(xy, axis=0))
        p1, p2 = _length_to_points(shaped.segment.length)
        edge = p2 - p1
        s1 = jnp.expand_dims(_vmap_dot(pb - p1, edge), axis=1)
        s2 = jnp.expand_dims(_vmap_dot(p2 - pb, edge), axis=1)
        in_segment = jnp.logical_and(s1 >= 0.0, s2 >= 0.0)
        ee = jnp.sum(jnp.square(edge), axis=-1, keepdims=True)
        pa = jnp.where(in_segment, p1 + edge * s1 / ee, jnp.where(s1 < 0.0, p1, p2))
        dist = jnp.linalg.norm(pb - pa, axis=-1)
        penetration = radius - dist
        has_overwrap = jnp.logical_and(stated.segment.is_active, penetration >= 0)
        overwrap2seg = jnp.any(has_overwrap)
    else:
        overwrap2seg = jnp.array(False)

    return jnp.logical_or(overwrap2cir, overwrap2seg)


def raycast_closest(
    fraction: float,
    p1: jax.Array,
    p2: jax.Array,
    shaped: ShapeDict,
    stated: StateDict,
):
    if shaped.circle is not None:
        rc = circle_raycast(0.0, fraction, p1, p2, shaped.circle, stated.circle)
        jnp.where(rc.hit, rc.fraction, 0.0)
    if shaped.static_circle is not None:
        rc = circle_raycast(
            0.0,
            fraction,
            p1,
            p2,
            shaped.static_circle,
            stated.static_circle,
        )
    if shaped.segment is not None:
        rc = segment_raycast(fraction, p1, p2, shaped.segment, stated.segment)
