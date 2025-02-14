"""Physics implementation"""

from __future__ import annotations

import dataclasses
import functools
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Generic, Protocol, TypeVar
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

Self = Any
T = TypeVar("T")
TWO_PI = jnp.pi * 2


def then(x: Any, f: Callable[[Any], Any]) -> Any:
    if x is None:
        return x
    else:
        return f(x)


def normalize(
    x: jax.Array,
    axis: Sequence[int] | int | None = None,
) -> tuple[jax.Array, jax.Array]:
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    n = x / jnp.clip(norm, min=1e-6)
    return n, jnp.squeeze(norm, axis=axis)


def _vmap_dot(xy1: jax.Array, xy2: jax.Array) -> jax.Array:
    """Dot product between nested vectors"""
    chex.assert_equal_shape((xy1, xy2))
    orig_shape = xy1.shape
    a = xy1.reshape(-1, orig_shape[-1])
    b = xy2.reshape(-1, orig_shape[-1])
    return jax.vmap(jnp.dot, in_axes=(0, 0))(a, b).reshape(*orig_shape[:-1])


def _sv_cross(s: jax.Array, v: jax.Array) -> jax.Array:
    """Cross product with scalar and vector"""
    x, y = _get_xy(v)
    return jnp.stack((y * -s, x * s), axis=-1)


def empty(cls: type[T]) -> Callable[[], T]:
    all_fields = {f.name: jnp.empty(0) for f in dataclasses.fields(cls)}  # type: ignore
    return lambda: cls(**all_fields)


class PyTreeOps:
    def __add__(self, o: Any) -> Self:
        if o.__class__ is self.__class__:
            return jax.tree_util.tree_map(lambda x, y: x + y, self, o)
        else:
            return jax.tree_util.tree_map(lambda x: x + o, self)

    def __sub__(self, o: Any) -> Self:
        if o.__class__ is self.__class__:
            return jax.tree_util.tree_map(lambda x, y: x - y, self, o)
        else:
            return jax.tree_util.tree_map(lambda x: x - o, self)

    def __mul__(self, o: float | jax.Array) -> Self:
        return jax.tree_util.tree_map(lambda x: x * o, self)

    def __neg__(self) -> Self:
        return jax.tree_util.tree_map(lambda x: -x, self)

    def __truediv__(self, o: float | jax.Array) -> Self:
        return jax.tree_util.tree_map(lambda x: x / o, self)

    @jax.jit
    def get_slice(self, index: Sequence[int] | int | None) -> Self:
        return jax.tree_util.tree_map(lambda x: x[index], self)

    def split(self, split_index: int) -> tuple[Self, Self]:
        def split_inner(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            return x[:split_index], x[split_index:]

        structure = jax.tree.structure(self)
        return jax.tree.transpose(
            structure,
            jax.tree.structure(("*", "*")),
            jax.tree.map(split_inner, self),
        )

    def reshape(self, shape: Sequence[int]) -> Self:
        return jax.tree_util.tree_map(lambda x: x.reshape(shape), self)

    def sum(self, axis: int | None = None) -> Self:
        return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=axis), self)

    def tolist(self) -> list[Self]:
        leaves, treedef = jax.tree_util.tree_flatten(self)
        return [treedef.unflatten(leaf) for leaf in zip(*leaves)]

    def zeros_like(self) -> Any:
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self)

    @property
    def shape(self) -> Any:
        """For debugging"""
        return jax.tree_util.tree_map(lambda x: x.shape, self)


def _axy(angle: jax.Array, xy: jax.Array) -> jax.Array:
    return jnp.concatenate((jnp.expand_dims(angle, axis=-1), xy), axis=-1)


class _PositionLike(Protocol):
    angle: jax.Array  # Angular velocity (N,)
    xy: jax.Array  # (N, 2)

    def __init__(self, angle: jax.Array, xy: jax.Array) -> None: ...

    def batch_size(self) -> int:
        return self.angle.shape[0]

    def into_axy(self) -> jax.Array:
        return _axy(self.angle, self.xy)

    @classmethod
    def zeros(cls: type[Self], n: int) -> Self:
        return cls(angle=jnp.zeros((n,)), xy=jnp.zeros((n, 2)))

    @classmethod
    def from_axy(cls: type[Self], axy: jax.Array) -> Self:
        angle = jax.lax.squeeze(jax.lax.slice_in_dim(axy, 0, 1, axis=-1), (-1,))
        xy = jax.lax.slice_in_dim(axy, 1, 3, axis=-1)
        return cls(angle=angle, xy=xy)


@chex.dataclass
class Velocity(_PositionLike, PyTreeOps):
    angle: jax.Array  # Angular velocity (N,)
    xy: jax.Array  # (N, 2)

    def rv(self, r: jax.Array) -> jax.Array:
        """Relative velocity"""
        return self.xy + _sv_cross(self.angle, r)


@chex.dataclass
class Force(_PositionLike, PyTreeOps):
    angle: jax.Array  # Angular (torque) force (N,)
    xy: jax.Array  # (N, 2)


def _get_xy(xy: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = jax.lax.slice_in_dim(xy, 0, 1, axis=-1)
    y = jax.lax.slice_in_dim(xy, 1, 2, axis=-1)
    return jax.lax.squeeze(x, (-1,)), jax.lax.squeeze(y, (-1,))


def _right_perp(xy: jax.Array) -> jax.Array:
    x = jax.lax.slice_in_dim(xy, 0, 1, axis=-1)
    y = jax.lax.slice_in_dim(xy, 1, 2, axis=-1)
    return jnp.concatenate((y, -x))


@chex.dataclass
class Position(_PositionLike, PyTreeOps):
    angle: jax.Array  # Angular velocity (N, 1)
    xy: jax.Array  # (N, 2)

    def rotate(self, xy: jax.Array) -> jax.Array:
        x, y = _get_xy(xy)
        s, c = jnp.sin(self.angle), jnp.cos(self.angle)
        rot_x = c * x - s * y
        rot_y = s * x + c * y
        return jnp.stack((rot_x, rot_y), axis=-1)

    def transform(self, xy: jax.Array) -> jax.Array:
        return self.rotate(xy) + self.xy

    def inv_rotate(self, xy: jax.Array) -> jax.Array:
        x, y = _get_xy(xy)
        s, c = jnp.sin(self.angle), jnp.cos(self.angle)
        rot_x = c * x + s * y
        rot_y = c * y - s * x
        return jnp.stack((rot_x, rot_y), axis=-1)

    def inv_transform(self, xy: jax.Array) -> jax.Array:
        return self.inv_rotate(xy - self.xy)

    def to_origin(self) -> Self:
        return replace(self, xy=jnp.zeros_like(self.xy))


@chex.dataclass
class Shape(PyTreeOps):
    mass: jax.Array
    moment: jax.Array
    elasticity: jax.Array
    friction: jax.Array
    rgba: jax.Array

    def batch_size(self) -> int:
        return self.mass.shape[0]

    def is_empty(self) -> bool:
        return self.batch_size() == 0

    def inv_mass(self) -> jax.Array:
        """To support static shape, set let inv_mass 0 if mass is infinite"""
        return jnp.where(jnp.isfinite(self.mass), 1.0 / self.mass, 0.0)

    def inv_moment(self) -> jax.Array:
        """As inv_mass does, set inv_moment 0 if moment is infinite"""
        return jnp.where(jnp.isfinite(self.moment), 1.0 / self.moment, 0.0)

    def to_shape(self) -> Self:
        return Shape(
            mass=self.mass,
            moment=self.moment,
            elasticity=self.elasticity,
            friction=self.friction,
            rgba=self.rgba,
        )


@chex.dataclass
class Circle(Shape):
    radius: jax.Array


@chex.dataclass
class Capsule(Shape):
    point1: jax.Array
    point2: jax.Array
    radius: jax.Array


@chex.dataclass
class Segment(Shape):
    point1: jax.Array
    point2: jax.Array
    is_smooth: jax.Array
    ghost1: jax.Array
    ghost2: jax.Array


@chex.dataclass
class Polygon(Shape):
    points: jax.Array
    radius: jax.Array
    normals: jax.Array
    centroid: jax.Array


@chex.dataclass
class Contact(PyTreeOps):
    pos: jax.Array
    normal: jax.Array
    penetration: jax.Array
    elasticity: jax.Array
    friction: jax.Array

    def contact_dim(self) -> int:
        return self.pos.shape[1]


@jax.vmap
def _circle_to_circle_impl(
    a: Circle,
    b: Circle,
    a_pos: Position,
    b_pos: Position,
    isactive: jax.Array,
) -> Contact:
    a2b_normal, dist = normalize(b_pos.xy - a_pos.xy)
    penetration = a.radius + b.radius - dist
    a_contact = a_pos.xy + a2b_normal * a.radius
    b_contact = b_pos.xy - a2b_normal * b.radius
    pos = (a_contact + b_contact) * 0.5
    penetration = jnp.where(isactive, penetration, -1.0)
    return Contact(
        pos=pos,
        normal=a2b_normal,
        penetration=penetration,
        elasticity=(a.elasticity + b.elasticity) * 0.5,
        friction=(a.friction + b.friction) * 0.5,
    )


@chex.dataclass
class ContactHelper:
    tangent: jax.Array
    mass_normal: jax.Array
    mass_tangent: jax.Array
    v_bias: jax.Array
    bounce: jax.Array
    r1: jax.Array
    r2: jax.Array
    inv_mass1: jax.Array
    inv_mass2: jax.Array
    inv_moment1: jax.Array
    inv_moment2: jax.Array
    local_anchor1: jax.Array
    local_anchor2: jax.Array
    allow_bounce: jax.Array


def _effective_mass(
    inv_mass: jax.Array,
    inv_moment: jax.Array,
    r: jax.Array,
    n: jax.Array,
) -> jax.Array:
    rn2 = jnp.cross(r, n) ** 2
    return inv_mass + inv_moment * rn2


@jax.vmap
def _capsule_to_circle_impl(
    a: Capsule,
    b: Circle,
    a_pos: Position,
    b_pos: Position,
    isactive: jax.Array,
) -> Contact:
    # Move b_pos to capsule's coordinates
    pb = a_pos.inv_transform(b_pos.xy)
    p1, p2 = a.point1, a.point2
    edge = p2 - p1
    s1 = jnp.dot(pb - p1, edge)
    s2 = jnp.dot(p2 - pb, edge)
    in_segment = jnp.logical_and(s1 >= 0.0, s2 >= 0.0)
    ee = jnp.sum(jnp.square(edge), axis=-1, keepdims=True)
    # Closest point
    # s1 < 0: pb is left to the capsule
    # s2 < 0: pb is right to the capsule
    # else: pb is in between capsule
    pa = jax.lax.select(
        in_segment,
        p1 + edge * s1 / ee,
        jax.lax.select(s1 < 0.0, p1, p2),
    )
    a2b_normal, dist = normalize(pb - pa)
    penetration = a.radius + b.radius - dist
    a_contact = pa + a2b_normal * a.radius
    b_contact = pb - a2b_normal * b.radius
    pos = a_pos.transform((a_contact + b_contact) * 0.5)
    a2b_normal_rotated = a_pos.to_origin().transform(a2b_normal)
    # Filter penetration
    return Contact(
        pos=pos,
        normal=a2b_normal_rotated,
        penetration=jnp.where(isactive, penetration, -1.0),
        elasticity=(a.elasticity + b.elasticity) * 0.5,
        friction=(a.friction + b.friction) * 0.5,
    )


@jax.vmap
def _segment_to_circle_impl(
    a: Segment,
    b: Circle,
    a_pos: Position,
    b_pos: Position,
    isactive: jax.Array,
) -> Contact:
    # Move b_pos to segment's coordinates
    pb = a_pos.inv_transform(b_pos.xy)
    p1, p2 = a.point1, a.point2
    edge = p2 - p1
    s1 = jnp.dot(pb - p1, edge)
    s2 = jnp.dot(p2 - pb, edge)
    in_segment = jnp.logical_and(s1 > 0.0, s2 > 0.0)
    ee = jnp.sum(jnp.square(edge), axis=-1, keepdims=True)
    # Closest point
    # s1 < 0: pb is left to the capsule
    # s2 < 0: pb is right to the capsule
    # else: pb is in between capsule
    pa = jax.lax.select(
        in_segment,
        p1 + edge * s1 / ee,
        jax.lax.select(s1 <= 0.0, p1, p2),
    )
    a2b_normal, dist = normalize(pb - pa)
    penetration = b.radius - dist
    a_contact = pa
    b_contact = pb - a2b_normal * b.radius
    pos = a_pos.transform((a_contact + b_contact) * 0.5)
    # Filter penetration
    collidable = jnp.dot(_right_perp(edge), pb - p1) >= 0.0
    not_in_voronoi = jnp.logical_or(
        jnp.logical_and(s1 <= 0.0, jnp.dot(a.ghost2 - p2, pb - p2) > 0.0),
        jnp.logical_and(s2 <= 0.0, jnp.dot(p1 - a.ghost1, pb - p1) <= 0.0),
    )
    is_penetration_possible = jnp.logical_and(
        isactive,
        jnp.logical_or(
            jnp.logical_not(a.is_smooth),
            # collidable
            jnp.logical_and(collidable, jnp.logical_not(not_in_voronoi)),
        ),
    )
    return Contact(
        pos=pos,
        normal=a_pos.to_origin().transform(a2b_normal),
        penetration=jnp.where(is_penetration_possible, penetration, -1.0),
        elasticity=(a.elasticity + b.elasticity) * 0.5,
        friction=(a.friction + b.friction) * 0.5,
    )


@jax.vmap
def _polygon_to_circle_impl(
    a: Polygon,
    b: Circle,
    a_pos: Position,
    b_pos: Position,
    isactive: jax.Array,
) -> Contact:
    n_vertices = a.points.shape[0]
    # Move circle pos to segment's coordinates
    c = a_pos.inv_transform(b_pos.xy)  # (1, 2)
    separation = _vmap_dot(a.normals, (c - a.points))
    max_sep = jnp.max(separation)
    i1 = jnp.argmax(separation)
    i2 = (i1 + 1) % n_vertices
    v1 = a.points[i1]
    v2 = a.points[i2]
    u1 = jnp.squeeze(jnp.dot(c - v1, v2 - v1))  # Convert to scalar
    u2 = jnp.squeeze(jnp.dot(c - v2, v1 - v2))

    def if_c_is_out() -> Contact:
        v = jax.lax.select(u1 < 0.0, v1, v2)
        a2b_normal, dist = normalize(c - v)
        ca = v + a.radius * a2b_normal
        cb = c - b.radius * a2b_normal
        pos_a = (ca + cb) * 0.5
        penetration = a.radius + b.radius - dist
        return Contact(
            pos=pos_a + a_pos.xy,
            normal=a_pos.to_origin().transform(a2b_normal),
            penetration=jnp.where(isactive, penetration, -1.0),
            elasticity=(a.elasticity + b.elasticity) * 0.5,
            friction=(a.friction + b.friction) * 0.5,
        )

    def if_c_is_in() -> Contact:
        a2b_normal = a.normals[i1]
        ca = c + a.radius - jnp.dot(c - v1, a2b_normal) * a2b_normal
        cb = c - b.radius * a2b_normal
        pos_a = (ca + cb) * 0.5
        penetration = a.radius + b.radius - max_sep
        return Contact(
            pos=pos_a + a_pos.xy,
            normal=a_pos.to_origin().transform(a2b_normal),
            penetration=jnp.where(isactive, penetration, -1.0),
            elasticity=(a.elasticity + b.elasticity) * 0.5,
            friction=(a.friction + b.friction) * 0.5,
        )

    is_out = jnp.logical_or(u1 < 0.0, u2 < 0.0)
    return jax.lax.cond(
        jnp.logical_and(is_out, max_sep > 1e-6),
        if_c_is_out,
        if_c_is_in,
    )


_ALL_SHAPES = [
    "circle",
    "static_circle",
    "capsule",
    "static_capsule",
    "segment",
    "triangle",
    "static_triangle",
    "quadrangle",
    "static_quadrangle",
    "pentagon",
    "static_pentagon",
    "hexagon",
    "static_hexagon",
]


@chex.dataclass
class State(PyTreeOps):
    p: Position
    v: Velocity
    f: Force
    is_active: jax.Array
    label: jax.Array

    @staticmethod
    def empty() -> Self:
        return State(
            p=Position.zeros(0),
            v=Velocity.zeros(0),
            f=Force.zeros(0),
            is_active=jnp.empty(0),
            label=jnp.zeros(0),
        )

    @staticmethod
    def zeros(n: int) -> Self:
        return State(
            p=Position.zeros(n),
            v=Velocity.zeros(n),
            f=Force.zeros(n),
            is_active=jnp.ones(n, dtype=bool),
            label=jnp.zeros(n, dtype=jnp.uint8),
        )

    def is_empty(self) -> bool:
        return self.p.batch_size() == 0

    def apply_force_global(self, point: jax.Array, force: jax.Array) -> Self:
        chex.assert_equal_shape((self.f.xy, force))
        xy = self.f.xy + force
        angle = self.f.angle + jnp.cross(point - self.p.xy, force)
        f = replace(self.f, xy=xy, angle=angle)
        return replace(self, f=f)

    def apply_force_local(self, point: jax.Array, force: jax.Array) -> Self:
        chex.assert_equal_shape((self.p.xy, point))
        point = self.p.transform(point)
        return self.apply_force_global(point, self.p.rotate(force))

    def batch_size(self) -> int:
        return self.p.batch_size()


def get_relative_angle(s_a: State, s_b: State) -> jax.Array:
    a2b = jax.vmap(jnp.subtract, in_axes=(None, 0))(s_b.p.xy, s_a.p.xy)
    a2b_x, a2b_y = _get_xy(a2b)
    a2b_angle = jnp.arctan2(a2b_y, a2b_x)  # (N_A, N_B)
    a_angle = jnp.expand_dims(s_a.p.angle, axis=1)
    # Subtract 0.5𝛑 because our angle starts from 0.5𝛑 (90 degree)
    return (a2b_angle - a_angle + TWO_PI * 3 - jnp.pi * 0.5) % TWO_PI


def _offset(sd: ShapeDict | StateDict, name: str) -> int:
    total = 0
    for key in _ALL_SHAPES:
        if key == name:
            return total
        s = sd[key]  # type: ignore
        if s is not None:
            total += s.batch_size()
    raise RuntimeError("Unreachable")


@chex.dataclass
class StateDict:
    circle: State = dataclasses.field(default_factory=State.empty)
    static_circle: State = dataclasses.field(default_factory=State.empty)
    segment: State = dataclasses.field(default_factory=State.empty)
    capsule: State = dataclasses.field(default_factory=State.empty)
    static_capsule: State = dataclasses.field(default_factory=State.empty)
    triangle: State = dataclasses.field(default_factory=State.empty)
    static_triangle: State = dataclasses.field(default_factory=State.empty)
    quadrangle: State = dataclasses.field(default_factory=State.empty)
    static_quadrangle: State = dataclasses.field(default_factory=State.empty)
    pentagon: State = dataclasses.field(default_factory=State.empty)
    static_pentagon: State = dataclasses.field(default_factory=State.empty)
    hexagon: State = dataclasses.field(default_factory=State.empty)
    static_hexagon: State = dataclasses.field(default_factory=State.empty)

    def concat(self) -> Self:
        states = [s for s in self.values() if s.batch_size() > 0]  # type: ignore
        return jax.tree_util.tree_map(
            lambda *args: jnp.concatenate(args, axis=0), *states
        )

    def update(self, statec: State) -> Self:
        def _get(name: str) -> State:
            state = self[name]  # type: ignore
            if state.is_empty():
                return state  # empty state
            else:
                start = _offset(self, name)
                end = start + state.p.batch_size()
                return statec.get_slice(jnp.arange(start, end))

        sd = {key: _get(key) for key in _ALL_SHAPES}
        return self.__class__(**sd)

    def nested_replace(self, query: str, value: Any) -> Self:
        """Convenient method for nested replace"""
        queries = query.split(".")
        objects = [self]
        for q in queries[:-1]:
            objects.append(objects[-1][q])  # type: ignore
        obj = replace(objects[-1], **{queries[-1]: value})
        for o, q in zip(objects[-2::-1], queries[-2::-1]):
            obj = replace(o, **{q: obj})
        return obj

    def nested_ops(self, query: str, fn: Callable[[Any], Any]) -> Self:
        queries = query.split(".")
        objects = [self]
        for q in queries[:-1]:
            objects.append(objects[-1][q])  # type: ignore
        value = fn(objects[-1][queries[-1]])  # type: ignore
        obj = replace(objects[-1], **{queries[-1]: value})
        for o, q in zip(objects[-2::-1], queries[-2::-1]):
            obj = replace(o, **{q: obj})
        return obj


@chex.dataclass
class ShapeDict:
    circle: Circle = dataclasses.field(default_factory=empty(Circle))
    static_circle: Circle = dataclasses.field(default_factory=empty(Circle))
    capsule: Capsule = dataclasses.field(default_factory=empty(Capsule))
    static_capsule: Capsule = dataclasses.field(default_factory=empty(Capsule))
    segment: Segment = dataclasses.field(default_factory=empty(Segment))
    triangle: Polygon = dataclasses.field(default_factory=empty(Polygon))
    static_triangle: Polygon = dataclasses.field(default_factory=empty(Polygon))
    quadrangle: Polygon = dataclasses.field(default_factory=empty(Polygon))
    static_quadrangle: Polygon = dataclasses.field(default_factory=empty(Polygon))
    pentagon: Polygon = dataclasses.field(default_factory=empty(Polygon))
    static_pentagon: Polygon = dataclasses.field(default_factory=empty(Polygon))
    hexagon: Polygon = dataclasses.field(default_factory=empty(Polygon))
    static_hexagon: Polygon = dataclasses.field(default_factory=empty(Polygon))

    def concat(self) -> Shape:
        shapes = [
            s.to_shape()
            for s in self.values()
            if s.batch_size() > 0  # type: ignore
        ]
        return jax.tree_util.tree_map(
            lambda *args: jnp.concatenate(args, axis=0), *shapes
        )

    def n_shapes(self) -> int:
        return sum([s.batch_size() for s in self.values()])  # type: ignore

    def zeros_state(self) -> StateDict:
        circle = then(self.circle, lambda s: State.zeros(len(s.mass)))
        static_circle = then(self.static_circle, lambda s: State.zeros(len(s.mass)))
        segment = then(self.segment, lambda s: State.zeros(len(s.mass)))
        capsule = then(self.capsule, lambda s: State.zeros(len(s.mass)))
        static_capsule = then(self.capsule, lambda s: State.zeros(len(s.mass)))
        triangle = then(self.triangle, lambda s: State.zeros(len(s.mass)))
        static_triangle = then(self.static_triangle, lambda s: State.zeros(len(s.mass)))
        quadrangle = then(self.quadrangle, lambda s: State.zeros(len(s.mass)))
        static_quadrangle = then(
            self.static_quadrangle,
            lambda s: State.zeros(len(s.mass)),
        )
        pentagon = then(self.pentagon, lambda s: State.zeros(len(s.mass)))
        static_pentagon = then(self.static_pentagon, lambda s: State.zeros(len(s.mass)))
        hexagon = then(self.hexagon, lambda s: State.zeros(len(s.mass)))
        static_hexagon = then(self.static_hexagon, lambda s: State.zeros(len(s.mass)))
        return StateDict(
            circle=circle,
            static_circle=static_circle,
            segment=segment,
            capsule=capsule,
            static_capsule=static_capsule,
            triangle=triangle,
            static_triangle=static_triangle,
            quadrangle=quadrangle,
            static_quadrangle=static_quadrangle,
            pentagon=pentagon,
            static_pentagon=static_pentagon,
            hexagon=hexagon,
            static_hexagon=static_hexagon,
        )


S1 = TypeVar("S1", bound=Shape)
S2 = TypeVar("S2", bound=Shape)


@chex.dataclass
class ContactIndices(Generic[S1, S2]):
    shape1: S1
    shape2: S2
    index1: jax.Array
    index2: jax.Array


def _self_ci(shape: Shape) -> ContactIndices:
    n = shape.batch_size()
    index1, index2 = jax.jit(jnp.triu_indices, static_argnums=(0, 1))(n, 1)
    return ContactIndices(
        shape1=shape.get_slice(index1),
        shape2=shape.get_slice(index2),
        index1=index1,
        index2=index2,
    )


def _pair_ci(shape1: Shape, shape2: Shape) -> ContactIndices:
    @functools.partial(jax.jit, static_argnums=(1,))
    def pair_outer(x: jax.Array, reps: int) -> jax.Array:
        return jnp.repeat(x, reps, axis=0, total_repeat_length=x.shape[0] * reps)

    @functools.partial(jax.jit, static_argnums=(1,))
    def pair_inner(x: jax.Array, reps: int) -> jax.Array:
        return jnp.tile(x, (reps,) + (1,) * (x.ndim - 1))

    n1, n2 = shape1.batch_size(), shape2.batch_size()
    index1 = pair_outer(jnp.arange(n1), reps=n2)
    index2 = pair_inner(jnp.arange(n2), reps=n1)
    return ContactIndices(
        shape1=shape1.get_slice(index1),
        shape2=shape2.get_slice(index2),
        index1=index1,
        index2=index2,
    )


def _circle_to_circle(ci: ContactIndices[Circle, Circle], stated: StateDict) -> Contact:
    return _circle_to_circle_impl(
        ci.shape1,
        ci.shape2,
        jax.tree_util.tree_map(lambda arr: arr[ci.index1], stated.circle.p),
        jax.tree_util.tree_map(lambda arr: arr[ci.index2], stated.circle.p),
        jnp.logical_and(
            stated.circle.is_active[ci.index1],
            stated.circle.is_active[ci.index2],
        ),
    )


def _circle_to_static_circle(
    ci: ContactIndices[Circle, Circle],
    stated: StateDict,
) -> Contact:
    return _circle_to_circle_impl(
        ci.shape1,
        ci.shape2,
        jax.tree_util.tree_map(lambda arr: arr[ci.index1], stated.circle.p),
        jax.tree_util.tree_map(lambda arr: arr[ci.index2], stated.static_circle.p),
        jnp.logical_and(
            stated.circle.is_active[ci.index1],
            stated.static_circle.is_active[ci.index2],
        ),
    )


def _capsule_to_circle(
    ci: ContactIndices[Capsule, Circle],
    stated: StateDict,
) -> Contact:
    return _capsule_to_circle_impl(
        ci.shape1,
        ci.shape2,
        jax.tree_util.tree_map(lambda arr: arr[ci.index1], stated.capsule.p),
        jax.tree_util.tree_map(lambda arr: arr[ci.index2], stated.circle.p),
        jnp.logical_and(
            stated.capsule.is_active[ci.index1],
            stated.circle.is_active[ci.index2],
        ),
    )


def _segment_to_circle(
    ci: ContactIndices[Segment, Circle],
    stated: StateDict,
) -> Contact:
    return _segment_to_circle_impl(
        ci.shape1,
        ci.shape2,
        jax.tree_util.tree_map(lambda arr: arr[ci.index1], stated.segment.p),
        jax.tree_util.tree_map(lambda arr: arr[ci.index2], stated.circle.p),
        jnp.logical_and(
            stated.segment.is_active[ci.index1], stated.circle.is_active[ci.index2]
        ),
    )


def _polygon_to_circle(
    ci: ContactIndices[Polygon, Circle],
    stated: StateDict,
    *,
    key: str,
) -> Contact:
    polygon = stated[key]  # type: ignore
    return _polygon_to_circle_impl(
        ci.shape1,
        ci.shape2,
        jax.tree_util.tree_map(lambda arr: arr[ci.index1], polygon.p),
        jax.tree_util.tree_map(lambda arr: arr[ci.index2], stated.circle.p),
        jnp.logical_and(
            polygon.is_active[ci.index1],
            stated.circle.is_active[ci.index2],
        ),
    )


def _polygon_to_polygon(
    ci: ContactIndices[Polygon, Polygon],
    stated: StateDict,
    *,
    key1: str,
    key2: str,
) -> Contact:
    assert False


_CONTACT_FN = Callable[[ContactIndices, StateDict], Contact]
_CONTACT_FUNCTIONS: dict[tuple[str, str], _CONTACT_FN] = {
    ("circle", "circle"): _circle_to_circle,
    ("circle", "static_circle"): _circle_to_static_circle,
    ("capsule", "circle"): _capsule_to_circle,
    ("segment", "circle"): _segment_to_circle,
    ("triangle", "circle"): functools.partial(_polygon_to_circle, key="triangle"),
    ("static_triangle", "circle"): functools.partial(
        _polygon_to_circle,
        key="static_triangle",
    ),
    # ("triangle", "triangle"): functools.partial(
    #     _polygon_to_polygon,
    #     key1="triangle",
    #     key2="triangle",
    # ),
}


# TODO: pretty print
@chex.dataclass
class Space:
    gravity: jax.Array
    shaped: ShapeDict
    dt: float = 0.1
    linear_damping: float = 0.95
    angular_damping: float = 0.95
    bias_factor: float = 0.2
    n_velocity_iter: int = 6
    n_position_iter: int = 2
    linear_slop: float = 0.005
    speculative_distance: float = 0.02
    max_linear_correction: float = 0.2
    allowed_penetration: float = 0.005
    bounce_threshold: float = 1.0
    max_velocity: float = 100.0
    max_angular_velocity: float = 100.0
    no_collision_constraints: dict[tuple[str, str], tuple[int, int]] = (
        dataclasses.field(default_factory=dict)
    )
    _contact_offset: dict[tuple[str, str], tuple[int, int]] = dataclasses.field(
        default_factory=dict,
        init=False,
    )
    _ci: dict[tuple[str, str], ContactIndices] = dataclasses.field(
        default_factory=dict,
        init=False,
    )
    _ci_total: ContactIndices = dataclasses.field(init=False)
    _ignore_flags: jax.Array = dataclasses.field(init=False)
    _hash_key: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, init=False)

    def __hash__(self) -> int:
        return hash(self._hash_key)

    def __eq__(self, other: Any) -> bool:
        return self._hash_key == other._hash_key

    def __post_init__(self) -> None:
        ci_slided_list = []
        offset = 0
        total_ci_len = 0
        for n1, n2 in _CONTACT_FUNCTIONS.keys():
            shape1, shape2 = self.shaped[n1], self.shaped[n2]  # type: ignore
            if shape1.batch_size() > 0 and shape2.batch_size() > 0:
                if n1 == n2:
                    ci = _self_ci(shape1)  # Type
                else:
                    ci = _pair_ci(shape1, shape2)
                self._ci[n1, n2] = ci
                total_ci_len += ci.index1.shape[0]
                offset_start = offset
                offset += ci.shape1.batch_size()
                self._contact_offset[n1, n2] = offset_start, offset
                offset1, offset2 = _offset(self.shaped, n1), _offset(self.shaped, n2)
                # Add some offset for global indices
                ci_slided = ContactIndices(
                    shape1=ci.shape1.to_shape(),
                    shape2=ci.shape2.to_shape(),
                    index1=ci.index1 + offset1,
                    index2=ci.index2 + offset2,
                )
                ci_slided_list.append(ci_slided)

        self._ci_total = jax.tree.map(
            lambda *args: jnp.concatenate(args, axis=0),
            *ci_slided_list,
        )
        self._ignore_flags = jnp.zeros(total_ci_len, dtype=bool)

    def set_ignore_flags_by_indices(
        self,
        target_n1: str,
        target_n2: str,
        n1_idx: Sequence[int] | int | None,
        n2_idx: Sequence[int] | int | None,
    ) -> None:
        start = 0
        for n1, n2 in _CONTACT_FUNCTIONS.keys():
            ci = self._ci.get((n1, n2), None)
            if ci is not None:
                n = ci.index1.shape[0]
                if target_n1 == n1 and target_n2 == n2:
                    false_array = jnp.zeros(n, dtype=bool)
                    flag1 = false_array.at[n1_idx].set(True)[ci.index1]
                    flag2 = false_array.at[n2_idx].set(True)[ci.index2]
                    self._ignore_flags = self._ignore_flags.at[start : start + n].set(
                        jnp.logical_and(flag1, flag2)
                    )
                    return
                else:
                    start += n

    def check_contacts(self, stated: StateDict) -> jax.Array:
        contacts = []
        for (n1, n2), fn in _CONTACT_FUNCTIONS.items():
            ci = self._ci.get((n1, n2), None)
            if ci is not None:
                contact = fn(ci, stated)
                contacts.append(contact)
        return jax.tree.map(
            lambda *args: jnp.concatenate(args, axis=0),
            *contacts,
        )

    def check_contacts_selected(
        self,
        stated: StateDict,
        selected: tuple[str, str],
    ) -> Contact | None:
        fn = _CONTACT_FUNCTIONS[selected]
        ci = self._ci.get(selected, None)
        if ci is not None:
            return fn(ci, stated)
        else:
            return None

    def n_possible_contacts(self) -> int:
        n = 0
        for n1, n2 in _CONTACT_FUNCTIONS.keys():
            shape1, shape2 = self.shaped[n1], self.shaped[n2]  # type: ignore
            len1, len2 = shape1.batch_size(), shape2.batch_size()
            if n1 == n2:
                n += len1 * (len1 - 1) // 2
            else:
                n += len1 * len2
        return n

    def get_contact_mat(self, n1: str, n2: str, contact: jax.Array) -> jax.Array:
        contact_offset = self._contact_offset.get((n1, n2), None)
        assert contact_offset is not None
        from_, to = contact_offset
        size1 = self.shaped[n1].batch_size()  # type: ignore
        size2 = self.shaped[n2].batch_size()  # type: ignore
        cnt = contact[from_:to]
        if n1 == n2:
            ret = jnp.zeros((size1, size1), dtype=bool)
            idx1, idx2 = jnp.triu_indices(size1, k=1)
            return ret.at[idx1, idx2].set(cnt).at[idx2, idx1].set(cnt)
        else:
            return contact[from_:to].reshape(size1, size2)

    def init_solver(self) -> VelocitySolver:
        n = self.n_possible_contacts()
        return VelocitySolver(
            v1=jnp.zeros((n, 3)),
            v2=jnp.zeros((n, 3)),
            pn=jnp.zeros(n),
            pt=jnp.zeros(n),
            contact=jnp.zeros(n, dtype=bool),
        )

    def zeros_state(self) -> StateDict:
        return self.shaped.zeros_state()


def update_velocity(space: Space, shape: Shape, state: State) -> State:
    # Expand (N, ) to (N, 1) because xy has a shape (N, 2)
    invm = jnp.expand_dims(shape.inv_mass(), axis=1)
    f_xy = jnp.where(
        jnp.logical_and(invm > 0, jnp.expand_dims(state.is_active, axis=1)),
        space.gravity * jnp.ones_like(state.v.xy) + state.f.xy * invm,
        jnp.zeros_like(state.v.xy),
    )
    v_xy = state.v.xy + f_xy * space.dt
    f_ang = jnp.where(state.is_active, state.f.angle, 0.0)
    v_ang = state.v.angle + f_ang * shape.inv_moment() * space.dt
    v_xy = jnp.clip(
        v_xy * space.linear_damping,
        min=-space.max_velocity,
        max=space.max_velocity,
    )
    v_ang = jnp.clip(
        v_ang * space.angular_damping,
        min=-space.max_angular_velocity,
        max=space.max_angular_velocity,
    )
    # Damping: dv/dt + vc = 0 -> v(t) = v0 * exp(-tc)
    # v(t + dt) = v0 * exp(-tc - dtc) = v0 * exp(-tc) * exp(-dtc) = v(t)exp(-dtc)
    # Thus, linear/angular damping factors are actually exp(-dtc)
    return replace(state, v=Velocity(angle=v_ang, xy=v_xy), f=state.f.zeros_like())


def update_position(space: Space, state: State) -> State:
    v_dt = state.v * space.dt
    xy = state.p.xy + v_dt.xy
    angle = (state.p.angle + v_dt.angle + TWO_PI) % TWO_PI
    return replace(state, p=Position(angle=angle, xy=xy))


def init_contact_helper(
    space: Space,
    contact: Contact,
    a: Shape,
    b: Shape,
    p1: Position,
    p2: Position,
    v1: Velocity,
    v2: Velocity,
) -> ContactHelper:
    r1 = contact.pos - p1.xy
    r2 = contact.pos - p2.xy

    inv_mass1, inv_mass2 = a.inv_mass(), b.inv_mass()
    inv_moment1, inv_moment2 = a.inv_moment(), b.inv_moment()
    kn1 = _effective_mass(inv_mass1, inv_moment1, r1, contact.normal)
    kn2 = _effective_mass(inv_mass2, inv_moment2, r2, contact.normal)
    nx, ny = _get_xy(contact.normal)
    tangent = jnp.stack((-ny, nx), axis=-1)
    kt1 = _effective_mass(inv_mass1, inv_moment1, r1, tangent)
    kt2 = _effective_mass(inv_mass2, inv_moment2, r2, tangent)
    clipped_p = jnp.clip(space.allowed_penetration - contact.penetration, max=0.0)
    v_bias = -space.bias_factor / space.dt * clipped_p
    # k_normal, k_tangent, and v_bias should have (N(N-1)/2, N_contacts) shape
    chex.assert_equal_shape((contact.friction, kn1, kn2, kt1, kt2, v_bias))
    # Compute elasiticity * relative_vel
    dv = v2.rv(r2) - v1.rv(r1)
    vn = _vmap_dot(dv, contact.normal)
    return ContactHelper(  # type: ignore
        tangent=tangent,
        mass_normal=1 / (kn1 + kn2),
        mass_tangent=1 / (kt1 + kt2),
        v_bias=v_bias,
        bounce=vn * contact.elasticity,
        r1=r1,
        r2=r2,
        inv_mass1=inv_mass1,
        inv_mass2=inv_mass2,
        inv_moment1=inv_moment1,
        inv_moment2=inv_moment2,
        local_anchor1=p1.inv_rotate(r1),
        local_anchor2=p2.inv_rotate(r2),
        allow_bounce=vn <= -space.bounce_threshold,
    )


@chex.dataclass
class VelocitySolver:
    v1: jax.Array
    v2: jax.Array
    pn: jax.Array
    pt: jax.Array
    contact: jax.Array

    def update(self, new_contact: jax.Array) -> Self:
        continuing_contact = jnp.logical_and(self.contact, new_contact)
        pn = jnp.where(continuing_contact, self.pn, 0.0)
        pt = jnp.where(continuing_contact, self.pt, 0.0)
        return replace(self, pn=pn, pt=pt, contact=new_contact)


@jax.vmap
def apply_initial_impulse(
    contact: Contact,
    helper: ContactHelper,
    solver: VelocitySolver,
) -> VelocitySolver:
    """Warm starting by applying initial impulse"""
    p = helper.tangent * solver.pt + contact.normal * solver.pn
    v1 = solver.v1 - _axy(
        angle=helper.inv_moment1 * jnp.cross(helper.r1, p),
        xy=p * helper.inv_mass1,
    )
    v2 = solver.v2 + _axy(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, p),
        xy=p * helper.inv_mass2,
    )
    return replace(solver, v1=v1, v2=v2)


def _rv_a2b(a: jax.Array, ra: jax.Array, b: jax.Array, rb: jax.Array):
    return Velocity.from_axy(b).rv(rb) - Velocity.from_axy(a).rv(ra)


@jax.vmap
def apply_velocity_normal(
    contact: Contact,
    helper: ContactHelper,
    solver: VelocitySolver,
) -> VelocitySolver:
    """
    Apply velocity constraints to the solver.
    Suppose that each shape has (N_contact, 1) or (N_contact, 2).
    """
    # Relative veclocity (from shape2 to shape1)
    dv = _rv_a2b(solver.v1, helper.r1, solver.v2, helper.r2)
    vt = jnp.dot(dv, helper.tangent)
    dpt = -helper.mass_tangent * vt
    # Clamp friction impulse
    max_pt = contact.friction * solver.pn
    pt = jnp.clip(solver.pt + dpt, min=-max_pt, max=max_pt)
    dpt_clamped = helper.tangent * (pt - solver.pt)
    # Velocity update by contact tangent
    dvt1 = _axy(
        angle=-helper.inv_moment1 * jnp.cross(helper.r1, dpt_clamped),
        xy=-dpt_clamped * helper.inv_mass1,
    )
    dvt2 = _axy(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, dpt_clamped),
        xy=dpt_clamped * helper.inv_mass2,
    )
    # Compute Relative velocity again
    dv = _rv_a2b(solver.v1 + dvt1, helper.r1, solver.v2 + dvt2, helper.r2)
    vn = _vmap_dot(dv, contact.normal)
    dpn = helper.mass_normal * (-vn + helper.v_bias)
    # Accumulate and clamp impulse
    pn = jnp.clip(solver.pn + dpn, min=0.0)
    dpn_clamped = contact.normal * (pn - solver.pn)
    # Velocity update by contact normal
    dvn1 = _axy(
        angle=-helper.inv_moment1 * jnp.cross(helper.r1, dpn_clamped),
        xy=-dpn_clamped * helper.inv_mass1,
    )
    dvn2 = _axy(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, dpn_clamped),
        xy=dpn_clamped * helper.inv_mass2,
    )
    # Filter dv
    return VelocitySolver(
        v1=jnp.where(solver.contact, dvn1 + dvt1, 0.0),
        v2=jnp.where(solver.contact, dvn2 + dvt2, 0.0),
        pn=pn,
        pt=pt,
        contact=solver.contact,
    )


@jax.vmap
def apply_bounce(
    contact: Contact,
    helper: ContactHelper,
    solver: VelocitySolver,
) -> tuple[jax.Array, jax.Array]:
    """
    Apply bounce (resititution).
    Suppose that each shape has (N_contact, 1) or (N_contact, 2).
    """
    # Relative veclocity (from shape2 to shape1)
    dv = _rv_a2b(solver.v1, helper.r1, solver.v2, helper.r2)
    vn = jnp.dot(dv, contact.normal)
    pn = -helper.mass_normal * (vn + helper.bounce)
    dpn = contact.normal * pn
    # Velocity update by contact normal
    dv1 = _axy(
        angle=-helper.inv_moment1 * jnp.cross(helper.r1, dpn),
        xy=-dpn * helper.inv_mass1,
    )
    dv2 = _axy(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, dpn),
        xy=dpn * helper.inv_mass2,
    )
    # Filter dv
    allow_bounce = jnp.logical_and(solver.contact, helper.allow_bounce)
    return jnp.where(allow_bounce, dv1, 0.0), jnp.where(allow_bounce, dv2, 0.0)


@chex.dataclass
class PositionSolver:
    p1: jax.Array
    p2: jax.Array
    contact: jax.Array
    min_separation: jax.Array


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0))
def correct_position(
    bias_factor: float | jax.Array,
    linear_slop: float | jax.Array,
    max_linear_correction: float | jax.Array,
    contact: Contact,
    helper: ContactHelper,
    solver: PositionSolver,
) -> PositionSolver:
    """
    Correct positions to remove penetration.
    Suppose that each shape in contact and helper has (N_contact, 1) or (N_contact, 2).
    p1 and p2 should have xy: (1, 2) angle (1, 1) shape
    """
    # (N_contact, 2)
    p1, p2 = Position.from_axy(solver.p1), Position.from_axy(solver.p2)
    r1 = p1.rotate(helper.local_anchor1)
    r2 = p2.rotate(helper.local_anchor2)
    ga2_ga1 = r2 - r1 + p2.xy - p1.xy
    separation = jnp.dot(ga2_ga1, contact.normal) - contact.penetration
    c = jnp.clip(
        bias_factor * (separation + linear_slop),
        min=-max_linear_correction,
        max=0.0,
    )
    kn1 = _effective_mass(helper.inv_mass1, helper.inv_moment1, r1, contact.normal)
    kn2 = _effective_mass(helper.inv_mass2, helper.inv_moment2, r2, contact.normal)
    k_normal = kn1 + kn2
    impulse = jnp.where(k_normal > 0.0, -c / k_normal, 0.0)
    pn = impulse * contact.normal
    dp1 = _axy(
        angle=-helper.inv_moment1 * jnp.cross(r1, pn),
        xy=-pn * helper.inv_mass1,
    )
    dp2 = _axy(
        angle=helper.inv_moment2 * jnp.cross(r2, pn),
        xy=pn * helper.inv_mass2,
    )
    min_sep = jnp.fmin(solver.min_separation, separation)
    # Filter p1/p2
    dp1 = jnp.where(solver.contact, dp1, 0.0)
    dp2 = jnp.where(solver.contact, dp2, 0.0)
    return replace(solver, p1=dp1, p2=dp2, min_separation=min_sep)


def solve_constraints(
    space: Space,
    solver: VelocitySolver,
    p: Position,
    v: Velocity,
    contact: Contact,
) -> tuple[Velocity, Position, VelocitySolver]:
    """Resolve collisions by Sequential Impulse method"""
    idx1, idx2 = space._ci_total.index1, space._ci_total.index2

    def gather(a: jax.Array, b: jax.Array, orig: jax.Array) -> jax.Array:
        return orig.at[idx1].add(a).at[idx2].add(b)

    p1, p2 = p.get_slice(idx1), p.get_slice(idx2)
    v1, v2 = v.get_slice(idx1), v.get_slice(idx2)
    helper = init_contact_helper(
        space,
        contact,
        space._ci_total.shape1,
        space._ci_total.shape2,
        p1,
        p2,
        v1,
        v2,
    )
    # Warm up the velocity solver
    solver = replace(solver, v1=v1.into_axy(), v2=v2.into_axy())
    solver = apply_initial_impulse(contact, helper, solver)

    def vstep(
        _: int,
        vs: tuple[jax.Array, VelocitySolver],
    ) -> tuple[jax.Array, VelocitySolver]:
        v_i, solver_i = vs
        solver_i1 = apply_velocity_normal(contact, helper, solver_i)
        v_i1 = gather(solver_i1.v1, solver_i1.v2, v_i)
        return v_i1, replace(solver_i1, v1=v_i1[idx1], v2=v_i1[idx2])

    v_axy, solver = jax.lax.fori_loop(
        0,
        space.n_velocity_iter,
        vstep,
        (v.into_axy(), solver),
    )
    bv1, bv2 = apply_bounce(contact, helper, solver)
    v_axy = gather(bv1, bv2, v_axy)

    def pstep(
        _: int,
        ps: tuple[jax.Array, PositionSolver],
    ) -> tuple[jax.Array, PositionSolver]:
        p_i, solver_i = ps
        solver_i1 = correct_position(
            space.bias_factor,
            space.linear_slop,
            space.max_linear_correction,
            contact,
            helper,
            solver_i,
        )
        p_i1 = gather(solver_i1.p1, solver_i1.p2, p_i)
        return p_i1, replace(solver_i1, p1=p_i1[idx1], p2=p_i1[idx2])

    pos_solver = PositionSolver(
        p1=p1.into_axy(),
        p2=p2.into_axy(),
        contact=solver.contact,
        min_separation=jnp.zeros_like(p1.angle),
    )
    p_axy, pos_solver = jax.lax.fori_loop(
        0,
        space.n_position_iter,
        pstep,
        (p.into_axy(), pos_solver),
    )
    return Velocity.from_axy(v_axy), Position.from_axy(p_axy), solver


def step(
    space: Space,
    stated: StateDict,
    solver: VelocitySolver,
) -> tuple[StateDict, VelocitySolver, Contact]:
    state = update_velocity(space, space.shaped.concat(), stated.concat())
    contact = space.check_contacts(stated.update(state))
    penetration_allowed = jnp.logical_and(
        contact.penetration >= space.speculative_distance,
        jnp.logical_not(space._ignore_flags),
    )
    v, p, solver = solve_constraints(
        space,
        solver.update(penetration_allowed),
        state.p,
        state.v,
        contact,
    )
    state = update_position(space, replace(state, v=v, p=p))
    return stated.update(state), solver, contact
