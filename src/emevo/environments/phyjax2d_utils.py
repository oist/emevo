import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from emevo.environments.phyjax2d import Capsule, Circle, Segment, ShapeDict, Space

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
        mass: float,
        moment: float,
        elasticity: float,
        rgba: Color = _BLACK,
    ) -> None:
        circle = Circle(
            radius=jnp.array([radius]),
            mass=jnp.array([mass]),
            moment=jnp.array([moment]),
            elasticity=jnp.array([elasticity]),
            rgba=jnp.array(rgba).reshape(1, 4),
        )
        self.circles.append(circle)

    def add_capsule(
        self,
        *,
        length: float,
        radius: float,
        mass: float,
        moment: float,
        elasticity: float,
        rgba: Color = _BLACK,
    ) -> None:
        capsule = Capsule(
            length=jnp.array([length]),
            radius=jnp.array([radius]),
            mass=jnp.array([mass]),
            moment=jnp.array([moment]),
            elasticity=jnp.array([elasticity]),
            rgba=jnp.array(rgba).reshape(1, 4),
        )
        self.capsules.append(capsule)

    def add_segment(
        self,
        *,
        length: float,
        mass: float,
        moment: float,
        elasticity: float,
        rgba: Color = _BLACK,
    ) -> None:
        segment = Segment(
            length=jnp.array([length]),
            mass=jnp.array([mass]),
            moment=jnp.array([moment]),
            elasticity=jnp.array([elasticity]),
            rgba=jnp.array(rgba).reshape(1, 4),
        )
        self.segments.append(segment)

    def build(self) -> Space:
        if len(self.circles) > 0:
            circle = jax.tree_map(lambda *args: jnp.stack(args), *self.circles)
        else:
            circle = None
        if len(self.capsules) > 0:
            capsule = jax.tree_map(lambda *args: jnp.stack(args), *self.capsules)
        else:
            capsule = None
        if len(self.segments) > 0:
            segment = jax.tree_map(lambda *args: jnp.stack(args), *self.segments)
        else:
            segment = None

        shaped = ShapeDict(
            circle=circle,
            segment=segment,
            capsule=capsule,
        )
        dt = self.dt
        linear_damping = jnp.exp(-dt * self.linear_damping)
        angular_damping = jnp.exp(-dt * self.angular_damping)
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
