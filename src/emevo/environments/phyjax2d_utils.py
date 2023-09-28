import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from emevo.environments.phyjax2d import (
    Capsule,
    Circle,
    Segment,
    Shape,
    ShapeDict,
    Space,
)

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
        return mass, moment


def _circle_mass(radius: float, density: float) -> tuple[jax.Array, jax.Array]:
    rr = radius**2
    mass = density * jnp.pi * rr
    moment = 0.5 * mass * rr
    return jnp.array([mass]), jax.array([moment])


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
    return jnp.array([mass]), jax.array([moment])


@dataclasses.dataclass
class SpaceBuilder:
    """
    A convenient builder for creating a space.
    Not expected to used with `jax.jit`.
    """

    gravity: tuple[float, float] = dataclasses.field(default=(0.0, -9.8))
    circles: list[Circle] = dataclasses.field(default_factory=list)
    capsules: list[Capsule] = dataclasses.field(default_factory=list)
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

    def add_circle(
        self,
        *,
        radius: float,
        density: float = 1.0,
        is_static: bool = False,
        friction: float = 0.8,
        elasticity: float = 0.8,
        rgba: Color = _BLACK,
    ) -> None:
        mass, moment = _mass_and_moment(*_circle_mass(radius, density), is_static)
        circle = Circle(
            radius=jnp.array([radius]),
            mass=mass,
            moment=moment,
            elasticity=jnp.array([elasticity]),
            friction=jnp.array([friction]),
            rgba=jnp.array(rgba).reshape(1, 4),
        )
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
        rgba: Color = _BLACK,
    ) -> None:
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
            rgba=jnp.array(rgba).reshape(1, 4),
        )
        self.capsules.append(capsule)

    def add_segment(
        self,
        *,
        length: float,
        friction: float = 0.8,
        elasticity: float = 0.8,
        rgba: Color = _BLACK,
    ) -> None:
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
        def stack_or(sl: list[Shape]) -> Shape | None:
            if len(sl) > 0:
                return jax.tree_map(lambda *args: jnp.stack(args), *sl)
            else:
                return None

        shaped = ShapeDict(
            circle=stack_or(self.circles),
            segment=stack_or(self.segments),
            capsule=stack_or(self.capsules),
        )
        dt = self.dt
        linear_damping = jnp.exp(-dt * self.linear_damping).item()
        angular_damping = jnp.exp(-dt * self.angular_damping).item()
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
        )
