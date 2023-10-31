import dataclasses
import functools
import uuid
from collections.abc import Sequence
from typing import Any, Callable, Protocol

import chex
import jax
import jax.numpy as jnp

Axis = Sequence[int] | int
Self = Any


def then(x: Any, f: Callable[[Any], Any]) -> Any:
    if x is None:
        return x
    else:
        return f(x)


def normalize(x: jax.Array, axis: Axis | None = None) -> tuple[jax.Array, jax.Array]:
    norm = jnp.linalg.norm(x, axis=axis)
    n = x / jnp.clip(norm, a_min=1e-6)
    return n, norm


def tree_map2(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[Any, Any]:
    """Same as tree_map, but returns a tuple"""
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    result = [f(*xs) for xs in zip(*all_leaves)]
    a = treedef.unflatten([elem[0] for elem in result])
    b = treedef.unflatten([elem[1] for elem in result])
    return a, b


def generate_self_pairs(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns two arrays that iterate over all combination of elements in x and y."""
    # x.shape[0] > 1
    chex.assert_axis_dimension_gt(x, 0, 1)
    n = x.shape[0]
    # (a, a, a, b, b, c)
    outer_loop = jnp.repeat(
        x,
        jnp.arange(n - 1, -1, -1),
        axis=0,
        total_repeat_length=n * (n - 1) // 2,
    )
    # (b, c, d, c, d, d)
    inner_loop = jnp.concatenate([x[i:] for i in range(1, len(x))])
    return outer_loop, inner_loop


def _pair_outer(x: jax.Array, reps: int) -> jax.Array:
    return jnp.repeat(x, reps, axis=0, total_repeat_length=x.shape[0] * reps)


def _pair_inner(x: jax.Array, reps: int) -> jax.Array:
    return jnp.tile(x, (reps,) + (1,) * (x.ndim - 1))


def generate_pairs(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns two arrays that iterate over all combination of elements in x and y"""
    xlen, ylen = x.shape[0], y.shape[0]
    return _pair_outer(x, ylen), _pair_inner(y, xlen)


class PyTreeOps:
    def __add__(self, o: Any) -> Self:
        if o.__class__ is self.__class__:
            return jax.tree_map(lambda x, y: x + y, self, o)
        else:
            return jax.tree_map(lambda x: x + o, self)

    def __sub__(self, o: Any) -> Self:
        if o.__class__ is self.__class__:
            return jax.tree_map(lambda x, y: x - y, self, o)
        else:
            return jax.tree_map(lambda x: x - o, self)

    def __mul__(self, o: float | jax.Array) -> Self:
        return jax.tree_map(lambda x: x * o, self)

    def __neg__(self) -> Self:
        return jax.tree_map(lambda x: -x, self)

    def __truediv__(self, o: float | jax.Array) -> Self:
        return jax.tree_map(lambda x: x / o, self)

    def get_slice(
        self,
        index: int | Sequence[int] | Sequence[bool] | jax.Array,
    ) -> Self:
        return jax.tree_map(lambda x: x[index], self)

    def reshape(self, shape: Sequence[int]) -> Self:
        return jax.tree_map(lambda x: x.reshape(shape), self)

    def sum(self, axis: int | None = None) -> Self:
        return jax.tree_map(lambda x: jnp.sum(x, axis=axis), self)

    def tolist(self) -> list[Self]:
        leaves, treedef = jax.tree_util.tree_flatten(self)
        return [treedef.unflatten(leaf) for leaf in zip(*leaves)]

    def zeros_like(self) -> Any:
        return jax.tree_map(lambda x: jnp.zeros_like(x), self)

    @property
    def shape(self) -> Any:
        """For debugging"""
        return jax.tree_map(lambda x: x.shape, self)


TWO_PI = jnp.pi * 2


def _axy(angle: jax.Array, xy: jax.Array) -> jax.Array:
    return jnp.concatenate((jnp.expand_dims(angle, axis=-1), xy), axis=-1)


class _PositionLike(Protocol):
    angle: jax.Array  # Angular velocity (N,)
    xy: jax.Array  # (N, 2)

    def __init__(self, angle: jax.Array, xy: jax.Array) -> None:
        ...

    def batch_size(self) -> int:
        return self.angle.shape[0]

    def into_axy(self) -> jax.Array:
        return _axy(self.angle, self.xy)

    @classmethod
    def zeros(cls: type[Self], n: int) -> Self:
        return cls(angle=jnp.zeros((n,)), xy=jnp.zeros((n, 2)))

    @classmethod
    def from_axy(cls: type[Self], axy: int) -> Self:
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


@chex.dataclass
class Shape(PyTreeOps):
    mass: jax.Array
    moment: jax.Array
    elasticity: jax.Array
    friction: jax.Array
    rgba: jax.Array

    def batch_size(self) -> int:
        return self.mass.shape[0]

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
class State(PyTreeOps):
    p: Position
    v: Velocity
    f: Force
    is_active: jax.Array

    @staticmethod
    def from_position(p: Position) -> Self:
        n = p.batch_size()
        return State(
            p=p,
            v=Velocity.zeros(n),
            f=Force.zeros(n),
            is_active=jnp.ones(n, dtype=bool),
        )

    @staticmethod
    def zeros(n: int) -> Self:
        return State(
            p=Position.zeros(n),
            v=Velocity.zeros(n),
            f=Force.zeros(n),
            is_active=jnp.ones(n, dtype=bool),
        )

    def apply_force_global(self, point: jax.Array, force: jax.Array) -> Self:
        chex.assert_equal_shape((self.f.xy, force))
        xy = self.f.xy + force
        angle = self.f.angle + jnp.cross(point - self.p.xy, force)
        f = self.f.replace(xy=xy, angle=angle)
        return self.replace(f=f)

    def apply_force_local(self, point: jax.Array, force: jax.Array) -> Self:
        chex.assert_equal_shape((self.p.xy, point))
        point = self.p.transform(point)
        return self.apply_force_global(point, self.p.rotate(force))

    def batch_size(self) -> int:
        return self.p.batch_size()


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
    # Filter penetration
    penetration = jnp.where(isactive, penetration, jnp.ones_like(penetration) * -1)
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
        return self.replace(pn=pn, pt=pt, contact=new_contact)


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


def _effective_mass(
    inv_mass: jax.Array,
    inv_moment: jax.Array,
    r: jax.Array,
    n: jax.Array,
) -> jax.Array:
    rn2 = jnp.cross(r, n) ** 2
    return inv_mass + inv_moment * rn2


@chex.dataclass
class Capsule(Shape):
    length: jax.Array
    radius: jax.Array


def _length_to_points(length: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = jnp.stack((length * -0.5, length * 0.0), axis=-1)
    b = jnp.stack((length * 0.5, length * 0.0), axis=-1)
    return a, b


@chex.dataclass
class Segment(Shape):
    length: jax.Array

    def to_capsule(self) -> Capsule:
        return Capsule(
            mass=self.mass,
            moment=self.moment,
            elasticity=self.elasticity,
            friction=self.friction,
            rgba=self.rgba,
            length=self.length,
            radius=jnp.zeros_like(self.length),
        )

    def get_ab(self) -> tuple[jax.Array, jax.Array]:
        return _length_to_points(self.length)


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
    p1, p2 = _length_to_points(a.length)
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
    xy_zeros = jnp.zeros_like(b_pos.xy)
    a2b_normal_rotated = a_pos.replace(xy=xy_zeros).transform(a2b_normal)
    # Filter penetration
    penetration = jnp.where(isactive, penetration, jnp.ones_like(penetration) * -1)
    return Contact(
        pos=pos,
        normal=a2b_normal_rotated,
        penetration=penetration,
        elasticity=(a.elasticity + b.elasticity) * 0.5,
        friction=(a.friction + b.friction) * 0.5,
    )


_ALL_SHAPES = ["circle", "static_circle", "capsule", "static_capsule", "segment"]


@chex.dataclass
class StateDict:
    circle: State | None = None
    static_circle: State | None = None
    segment: State | None = None
    capsule: State | None = None
    static_capsule: State | None = None

    def concat(self) -> Self:
        states = [s for s in self.values() if s is not None]
        return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *states)

    def _get(self, name: str, state: State) -> State | None:
        if self[name] is None:
            return None
        else:
            start = _offset(self, name)
            end = start + self[name].p.batch_size()
            return state.get_slice(jnp.arange(start, end))

    def update(self, statec: State) -> Self:
        circle = self._get("circle", statec)
        static_circle = self._get("static_circle", statec)
        segment = self._get("segment", statec)
        capsule = self._get("capsule", statec)
        static_capsule = self._get("static_capsule", statec)
        return self.__class__(
            circle=circle,
            static_circle=static_circle,
            segment=segment,
            capsule=capsule,
            static_capsule=static_capsule,
        )

    @functools.partial(jax.jit, static_argnums=(1,))
    def nested_replace(self, query: str, value: Any) -> Self:
        """Convenient method for nested replace"""
        queries = query.split(".")
        objects = [self]
        for q in queries[:-1]:
            objects.append(objects[-1][q])  # type: ignore
        obj = objects[-1].replace(**{queries[-1]: value})  # type: ignore
        for o, q in zip(objects[-2::-1], queries[-2::-1]):
            obj = o.replace(**{q: obj})  # type: ignore
        return obj


@chex.dataclass
class ShapeDict:
    circle: Circle | None = None
    static_circle: Circle | None = None
    capsule: Capsule | None = None
    static_capsule: Capsule | None = None
    segment: Segment | None = None

    def concat(self) -> Shape:
        shapes = [s.to_shape() for s in self.values() if s is not None]
        return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *shapes)

    def zeros_state(self) -> StateDict:
        circle = then(self.circle, lambda s: State.zeros(len(s.mass)))
        static_circle = then(self.static_circle, lambda s: State.zeros(len(s.mass)))
        segment = then(self.segment, lambda s: State.zeros(len(s.mass)))
        capsule = then(self.capsule, lambda s: State.zeros(len(s.mass)))
        static_capsule = then(self.capsule, lambda s: State.zeros(len(s.mass)))
        return StateDict(
            circle=circle,
            static_circle=static_circle,
            segment=segment,
            capsule=capsule,
            static_capsule=static_capsule,
        )


def _offset(sd: ShapeDict | StateDict, name: str) -> int:
    total = 0
    for key in _ALL_SHAPES:
        if key == name:
            return total
        s = sd[key]
        if s is not None:
            total += s.batch_size()
    raise RuntimeError("Unreachable")


@chex.dataclass
class ContactIndices:
    shape1: Shape
    shape2: Shape
    index1: jax.Array
    index2: jax.Array


# These fuctions are used in __post_init__ so need to jit
_jitted_self_pairs = jax.jit(generate_self_pairs)
_jitted_pairs = jax.jit(generate_pairs)
_jitted_pair_outer = jax.jit(_pair_outer, static_argnums=(1,))
_jitted_pair_inner = jax.jit(_pair_inner, static_argnums=(1,))


def _self_ci(shape: Shape) -> ContactIndices:
    shape1, shape2 = tree_map2(_jitted_self_pairs, shape)
    index1, index2 = _jitted_self_pairs(jnp.arange(shape.mass.shape[0]))
    return ContactIndices(
        shape1=shape1,
        shape2=shape2,
        index1=index1,
        index2=index2,
    )


def _pair_ci(shape1: Shape, shape2: Shape) -> ContactIndices:
    n1, n2 = shape1.mass.shape[0], shape2.mass.shape[0]
    s1_extended = jax.tree_map(functools.partial(_jitted_pair_outer, reps=n2), shape1)
    s2_extended = jax.tree_map(functools.partial(_jitted_pair_inner, reps=n1), shape2)
    index1, index2 = _jitted_pairs(jnp.arange(n1), jnp.arange(n2))
    return ContactIndices(
        shape1=s1_extended,
        shape2=s2_extended,
        index1=index1,
        index2=index2,
    )


def _circle_to_circle(ci: ContactIndices, stated: StateDict) -> Contact:
    pos1 = jax.tree_map(lambda arr: arr[ci.index1], stated.circle.p)
    pos2 = jax.tree_map(lambda arr: arr[ci.index2], stated.circle.p)
    is_active1 = stated.circle.is_active[ci.index1]
    is_active2 = stated.circle.is_active[ci.index2]
    return _circle_to_circle_impl(
        ci.shape1,
        ci.shape2,
        pos1,
        pos2,
        jnp.logical_and(is_active1, is_active2),
    )


def _circle_to_static_circle(ci: ContactIndices, stated: StateDict) -> Contact:
    pos1 = jax.tree_map(lambda arr: arr[ci.index1], stated.circle.p)
    pos2 = jax.tree_map(lambda arr: arr[ci.index2], stated.static_circle.p)
    is_active1 = stated.circle.is_active[ci.index1]
    is_active2 = stated.static_circle.is_active[ci.index2]
    return _circle_to_circle_impl(
        ci.shape1,
        ci.shape2,
        pos1,
        pos2,
        jnp.logical_and(is_active1, is_active2),
    )


def _capsule_to_circle(ci: ContactIndices, stated: StateDict) -> Contact:
    pos1 = jax.tree_map(lambda arr: arr[ci.index1], stated.capsule.p)
    pos2 = jax.tree_map(lambda arr: arr[ci.index2], stated.circle.p)
    is_active1 = stated.capsule.is_active[ci.index1]
    is_active2 = stated.circle.is_active[ci.index2]
    return _circle_to_circle_impl(
        ci.shape1,
        ci.shape2,
        pos1,
        pos2,
        jnp.logical_and(is_active1, is_active2),
    )


def _segment_to_circle(ci: ContactIndices, stated: StateDict) -> Contact:
    pos1 = jax.tree_map(lambda arr: arr[ci.index1], stated.segment.p)
    pos2 = jax.tree_map(lambda arr: arr[ci.index2], stated.circle.p)
    is_active1 = stated.segment.is_active[ci.index1]
    is_active2 = stated.circle.is_active[ci.index2]
    return _capsule_to_circle_impl(
        ci.shape1.to_capsule(),
        ci.shape2,
        pos1,
        pos2,
        jnp.logical_and(is_active1, is_active2),
    )


_CONTACT_FUNCTIONS = {
    ("circle", "circle"): _circle_to_circle,
    ("circle", "static_circle"): _circle_to_static_circle,
    ("capsule", "circle"): _capsule_to_circle,
    ("segment", "circle"): _segment_to_circle,
}


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
    max_linear_correction: float = 0.2
    allowed_penetration: float = 0.005
    bounce_threshold: float = 1.0
    max_velocity: float = 100.0
    max_angular_velocity: float = 100.0
    _ci: dict[tuple[str, str], ContactIndices] = dataclasses.field(
        default_factory=dict,
        init=False,
    )
    _ci_total: ContactIndices | None = dataclasses.field(default=None, init=False)
    _hash_key: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, init=False)

    def __hash__(self) -> int:
        return hash(self._hash_key)

    def __eq__(self, other: Any) -> int:
        return self._hash_key == other._hash_key

    def __post_init__(self) -> None:
        ci_slided_list = []
        for n1, n2 in _CONTACT_FUNCTIONS.keys():
            if self.shaped[n1] is not None and self.shaped[n2] is not None:
                if n1 == n2:
                    ci = _self_ci(self.shaped[n1])
                else:
                    ci = _pair_ci(self.shaped[n1], self.shaped[n2])
                self._ci[n1, n2] = ci
                offset1, offset2 = _offset(self.shaped, n1), _offset(self.shaped, n2)
                # Add some offset for global indices
                ci_slided = ContactIndices(
                    shape1=ci.shape1.to_shape(),
                    shape2=ci.shape2.to_shape(),
                    index1=ci.index1 + offset1,
                    index2=ci.index2 + offset2,
                )
                ci_slided_list.append(ci_slided)
        self._ci_total = jax.tree_map(
            lambda *args: jnp.concatenate(args, axis=0),
            *ci_slided_list,
        )

    def check_contacts(self, stated: StateDict) -> Contact:
        contacts = []
        for (n1, n2), fn in _CONTACT_FUNCTIONS.items():
            ci = self._ci.get((n1, n2), None)
            if ci is not None:
                contacts.append(fn(ci, stated))
        return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *contacts)

    def n_possible_contacts(self) -> int:
        n = 0
        for n1, n2 in _CONTACT_FUNCTIONS.keys():
            if self.shaped[n1] is not None and self.shaped[n2] is not None:
                len1, len2 = len(self.shaped[n1].mass), len(self.shaped[n2].mass)
                if n1 == n2:
                    n += len1 * (len1 - 1) // 2
                else:
                    n += len1 * len2
        return n

    def init_solver(self) -> VelocitySolver:
        n = self.n_possible_contacts()
        return VelocitySolver(
            v1=jnp.zeros((n, 3)),
            v2=jnp.zeros((n, 3)),
            pn=jnp.zeros(n),
            pt=jnp.zeros(n),
            contact=jnp.zeros(n, dtype=bool),
        )


def update_velocity(space: Space, shape: Shape, state: State) -> State:
    # Expand (N, ) to (N, 1) because xy has a shape (N, 2)
    invm = jnp.expand_dims(shape.inv_mass(), axis=1)
    gravity = jnp.where(
        jnp.logical_and(invm > 0, jnp.expand_dims(state.is_active, axis=1)),
        space.gravity * jnp.ones_like(state.v.xy),
        jnp.zeros_like(state.v.xy),
    )
    v_xy = state.v.xy + (gravity + state.f.xy * invm) * space.dt
    v_ang = state.v.angle + state.f.angle * shape.inv_moment() * space.dt
    v_xy = jnp.clip(
        v_xy * space.linear_damping,
        a_max=space.max_velocity,
        a_min=-space.max_velocity,
    )
    v_ang = jnp.clip(
        v_ang * space.angular_damping,
        a_max=space.max_angular_velocity,
        a_min=-space.max_angular_velocity,
    )
    # Damping: dv/dt + vc = 0 -> v(t) = v0 * exp(-tc)
    # v(t + dt) = v0 * exp(-tc - dtc) = v0 * exp(-tc) * exp(-dtc) = v(t)exp(-dtc)
    # Thus, linear/angular damping factors are actually exp(-dtc)
    return state.replace(v=Velocity(angle=v_ang, xy=v_xy), f=state.f.zeros_like())


def update_position(space: Space, state: State) -> State:
    v_dt = state.v * space.dt
    xy = state.p.xy + v_dt.xy
    angle = (state.p.angle + v_dt.angle + TWO_PI) % TWO_PI
    return state.replace(p=Position(angle=angle, xy=xy))


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
    clipped_p = jnp.clip(space.allowed_penetration - contact.penetration, a_max=0.0)
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
    return solver.replace(v1=v1, v2=v2)


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
    pt = jnp.clip(solver.pt + dpt, a_min=-max_pt, a_max=max_pt)
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
    pn = jnp.clip(solver.pn + dpn, a_min=0.0)
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
        a_min=-max_linear_correction,
        a_max=0.0,
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
    return solver.replace(p1=dp1, p2=dp2, min_separation=min_sep)


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
    solver = solver.replace(v1=v1.into_axy(), v2=v2.into_axy())
    solver = apply_initial_impulse(contact, helper, solver)

    def vstep(
        _n_iter: int,
        vs: tuple[Velocity, VelocitySolver],
    ) -> tuple[Velocity, VelocitySolver]:
        v_i, solver_i = vs
        solver_i1 = apply_velocity_normal(contact, helper, solver_i)
        v_i1 = gather(solver_i1.v1, solver_i1.v2, v_i)
        return v_i1, solver_i1.replace(v1=v_i1[idx1], v2=v_i1[idx2])

    v, solver = jax.lax.fori_loop(
        0,
        space.n_velocity_iter,
        vstep,
        (v.into_axy(), solver),
    )
    bv1, bv2 = apply_bounce(contact, helper, solver)
    v = gather(bv1, bv2, v)

    def pstep(
        _n_iter: int,
        ps: tuple[Position, PositionSolver],
    ) -> tuple[Position, PositionSolver]:
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
        return p_i1, solver_i1.replace(p1=p_i1[idx1], p2=p_i1[idx2])

    pos_solver = PositionSolver(
        p1=p1.into_axy(),
        p2=p2.into_axy(),
        contact=solver.contact,
        min_separation=jnp.zeros_like(p1.angle),
    )
    p, pos_solver = jax.lax.fori_loop(
        0,
        space.n_position_iter,
        pstep,
        (p.into_axy(), pos_solver),
    )
    return Velocity.from_axy(v), Position.from_axy(p), solver


def step(
    space: Space,
    stated: StateDict,
    solver: VelocitySolver,
) -> tuple[StateDict, VelocitySolver]:
    state = update_velocity(space, space.shaped.concat(), stated.concat())
    contact = space.check_contacts(stated.update(state))
    v, p, solver = solve_constraints(
        space,
        solver.update(contact.penetration >= 0),
        state.p,
        state.v,
        contact,
    )
    state = update_position(space, state.replace(v=v, p=p))
    return stated.update(state), solver


def nstep(
    n: int,
    space: Space,
    stated: StateDict,
    solver: VelocitySolver,
) -> tuple[StateDict, VelocitySolver]:
    def wrapped_step(
        _n_iter: int,
        stated_and_solver: tuple[StateDict, VelocitySolver],
    ) -> tuple[StateDict, VelocitySolver]:
        return step(space, *stated_and_solver)

    return jax.lax.fori_loop(0, n, wrapped_step, (stated, solver))


@chex.dataclass
class Raycast:
    fraction: jax.Array
    normal: jax.Array
    hit: jax.Array


def circle_raycast(
    radius: float | jax.Array,
    max_fraction: float | jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    circle: Circle,
    state: State,
) -> Raycast:
    # Suppose p1 and p2's shape has (2,)
    s = jnp.expand_dims(p1, axis=0) - state.p.xy  # (N, 2)
    d, length = normalize(p2 - p1)
    t = -jnp.dot(s, d)  # (N,)

    @jax.vmap
    def muld(x: jax.Array) -> jax.Array:
        return x * d

    c = s + muld(t)  # (N, 2)
    cc = _vmap_dot(c, c)  # (N, 1)
    rr = (radius + circle.radius) ** 2
    fraction = jnp.where(rr >= cc, t - jnp.sqrt(rr - cc), 0.0)
    hitpoint = s + muld(fraction)
    normal, _ = normalize(hitpoint)
    return Raycast(  # type: ignore
        fraction=fraction / length,
        normal=normal,
        hit=jnp.logical_and(
            cc <= rr,
            jnp.logical_and(0.0 <= fraction, fraction <= max_fraction * length),
        ),
    )


def segment_raycast(
    max_fraction: float | jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    segment: Segment,
    state: State,
) -> Raycast:
    d = p2 - p1
    v1, v2 = _length_to_points(segment.length)
    v1, v2 = state.p.transform(v1), state.p.transform(v2)
    e = v2 - v1
    eunit, length = normalize(e)
    normal = _sv_cross(jnp.ones_like(length) * -1, eunit)
    numerator = _vmap_dot(normal, v1 - p1)  # (N,)
    denominator = jnp.dot(normal, d)  # (N,)
    t = numerator / denominator
    p = jax.vmap(lambda ti: ti * d + p1)(t)  # (N, 2)
    s = _vmap_dot(p - v1, eunit)
    normal = jnp.where(jnp.expand_dims(numerator > 0.0, axis=1), -normal, normal)
    return Raycast(  # type: ignore
        fraction=t,
        normal=normal,
        hit=jnp.logical_and(
            denominator != 0.0,
            jnp.logical_and(
                jnp.logical_and(t >= 0.0, t <= max_fraction),
                jnp.logical_and(s >= 0.0, s <= length),
            ),
        ),
    )
