import functools
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


def safe_norm(x: jax.Array, axis: Axis | None = None) -> jax.Array:
    is_zero = jnp.allclose(x, 0.0)
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis)
    return jnp.where(is_zero, 0.0, n)  # pyright: ignore


def normalize(x: jax.Array, axis: Axis | None = None) -> tuple[jax.Array, jax.Array]:
    norm = safe_norm(x, axis=axis)
    n = x / (norm + 1e-6 * (norm == 0.0))
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


class _PositionLike(Protocol):
    angle: jax.Array  # Angular velocity (N,)
    xy: jax.Array  # (N, 2)

    def __init__(self, angle: jax.Array, xy: jax.Array) -> None:
        ...

    def batch_size(self) -> int:
        return self.angle.shape[0]

    @classmethod
    def zeros(cls: type[Self], n: int) -> Self:
        return cls(angle=jnp.zeros((n,)), xy=jnp.zeros((n, 2)))


@chex.dataclass
class Velocity(_PositionLike, PyTreeOps):
    angle: jax.Array  # Angular velocity (N,)
    xy: jax.Array  # (N, 2)


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

    def inv_mass(self) -> jax.Array:
        """To support static shape, set let inv_mass 0 if mass is infinite"""
        m = self.mass
        return jnp.where(jnp.isfinite(m), 1.0 / m, jnp.zeros_like(m))

    def inv_moment(self) -> jax.Array:
        """As inv_mass does, set inv_moment 0 if moment is infinite"""
        m = self.moment
        return jnp.where(jnp.isfinite(m), 1.0 / m, jnp.zeros_like(m))

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
        return State(p=p, v=Velocity.zeros(n), f=Force.zeros(n), is_active=jnp.ones(n))

    @staticmethod
    def zeros(n: int) -> Self:
        return State(
            p=Position.zeros(n),
            v=Velocity.zeros(n),
            f=Force.zeros(n),
            is_active=jnp.ones(n),
        )


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
    v1: Velocity
    v2: Velocity
    pn: jax.Array
    pt: jax.Array
    contact: jax.Array

    def update(self, new_contact: jax.Array) -> Self:
        continuing_contact = jnp.logical_and(self.contact, new_contact)
        pn = jnp.where(continuing_contact, self.pn, jnp.zeros_like(self.pn))
        pt = jnp.where(continuing_contact, self.pt, jnp.zeros_like(self.pt))
        return self.replace(pn=pn, pt=pt, contact=new_contact)


def init_solver(n: int) -> VelocitySolver:
    return VelocitySolver(
        v1=Velocity.zeros(n),
        v2=Velocity.zeros(n),
        pn=jnp.zeros(n),
        pt=jnp.zeros(n),
        contact=jnp.zeros(n, dtype=bool),
    )


def _pv_gather(
    p1: _PositionLike,
    p2: _PositionLike,
    orig: _PositionLike,
) -> _PositionLike:
    indices = jnp.arange(len(orig.angle))
    outer, inner = generate_self_pairs(indices)
    p1_xy = jnp.zeros_like(orig.xy).at[outer].add(p1.xy)
    p1_angle = jnp.zeros_like(orig.angle).at[outer].add(p1.angle)
    p2_xy = jnp.zeros_like(orig.xy).at[inner].add(p2.xy)
    p2_angle = jnp.zeros_like(orig.angle).at[inner].add(p2.angle)
    return p1.__class__(xy=p1_xy + p2_xy, angle=p1_angle + p2_angle)


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


def _dv2from1(v1: Velocity, r1: jax.Array, v2: Velocity, r2: jax.Array) -> jax.Array:
    """Compute relative veclotiy from v2/r2 to v1/r1"""
    rel_v1 = v1.xy + _sv_cross(v1.angle, r1)
    rel_v2 = v2.xy + _sv_cross(v2.angle, r2)
    return rel_v2 - rel_v1


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


def _length_to_points(length: jax.Array) -> tuple[jax.Array, jax.Array]:
    a = jnp.stack((length * -0.5, length * 0.0), axis=-1)
    b = jnp.stack((length * 0.5, length * 0.0), axis=-1)
    return a, b


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


@chex.dataclass
class StateDict:
    circle: State | None = None
    segment: State | None = None
    capsule: State | None = None

    def concat(self) -> Self:
        states = [s for s in self.values() if s is not None]
        return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *states)

    def offset(self, key: str) -> int:
        total = 0
        for k, state in self.items():
            if k == key:
                return total
            if state is not None:
                total += state.p.batch_size()
        raise RuntimeError("Unreachable")

    def _get(self, name: str, state: State) -> State | None:
        if self[name] is None:
            return None
        else:
            start = self.offset(name)
            end = start + self[name].p.batch_size()
            return state.get_slice(jnp.arange(start, end))

    def update(self, statec: State) -> Self:
        circle = self._get("circle", statec)
        segment = self._get("segment", statec)
        capsule = self._get("capsule", statec)
        return self.__class__(circle=circle, segment=segment, capsule=capsule)


@chex.dataclass
class ShapeDict:
    circle: Circle | None = None
    segment: Segment | None = None
    capsule: Capsule | None = None

    def concat(self) -> Shape:
        shapes = [s.to_shape() for s in self.values() if s is not None]
        return jax.tree_map(lambda *args: jnp.concatenate(args, axis=0), *shapes)

    def zeros_state(self) -> StateDict:
        circle = then(self.circle, lambda s: State.zeros(len(s.mass)))
        segment = then(self.segment, lambda s: State.zeros(len(s.mass)))
        capsule = then(self.capsule, lambda s: State.zeros(len(s.mass)))
        return StateDict(circle=circle, segment=segment, capsule=capsule)


def _circle_to_circle(
    shaped: ShapeDict,
    stated: StateDict,
) -> tuple[Contact, Circle, Circle]:
    circle1, circle2 = tree_map2(generate_self_pairs, shaped.circle)
    pos1, pos2 = tree_map2(generate_self_pairs, stated.circle.p)
    is_active = jnp.logical_and(*generate_self_pairs(stated.circle.is_active))
    contacts = _circle_to_circle_impl(
        circle1,
        circle2,
        pos1,
        pos2,
        is_active,
    )
    return contacts, circle1, circle2


def _capsule_to_circle(
    shaped: ShapeDict,
    stated: StateDict,
) -> tuple[Contact, Capsule, Circle]:
    capsule = jax.tree_map(
        functools.partial(_pair_outer, reps=shaped.circle.mass.shape[0]),
        shaped.capsule,
    )
    circle = jax.tree_map(
        functools.partial(_pair_inner, reps=shaped.capsule.mass.shape[0]),
        shaped.circle,
    )
    pos1, pos2 = tree_map2(generate_pairs, stated.capsule.p, stated.circle.p)
    is_active = jnp.logical_and(
        *generate_pairs(stated.capsule.is_active, stated.circle.is_active)
    )
    contacts = _capsule_to_circle_impl(
        capsule,
        circle,
        pos1,
        pos2,
        is_active,
    )
    return contacts, capsule, circle


def _segment_to_circle(
    shaped: ShapeDict,
    stated: StateDict,
) -> tuple[Contact, Segment, Circle]:
    segment = jax.tree_map(
        functools.partial(_pair_outer, reps=shaped.circle.mass.shape[0]),
        shaped.segment,
    )
    circle = jax.tree_map(
        functools.partial(_pair_inner, reps=shaped.segment.mass.shape[0]),
        shaped.circle,
    )
    pos1, pos2 = tree_map2(generate_pairs, stated.segment.p, stated.circle.p)
    is_active = jnp.logical_and(
        *generate_pairs(stated.segment.is_active, stated.circle.is_active)
    )
    contacts = _capsule_to_circle_impl(
        segment.to_capsule(),
        circle,
        pos1,
        pos2,
        is_active,
    )
    return contacts, segment, circle


_CONTACT_FUNCTIONS = {
    ("circle", "circle"): _circle_to_circle,
    ("capsule", "circle"): _capsule_to_circle,
    ("segment", "circle"): _segment_to_circle,
}


@chex.dataclass
class ContactWithMetadata:
    contact: Contact
    shape1: Shape
    shape2: Shape
    outer_index: jax.Array
    inner_index: jax.Array

    def gather_p_or_v(
        self,
        outer: _PositionLike,
        inner: _PositionLike,
        orig: _PositionLike,
    ) -> _PositionLike:
        xy_outer = jnp.zeros_like(orig.xy).at[self.outer_index].add(outer.xy)
        angle_outer = jnp.zeros_like(orig.angle).at[self.outer_index].add(outer.angle)
        xy_inner = jnp.zeros_like(orig.xy).at[self.inner_index].add(inner.xy)
        angle_inner = jnp.zeros_like(orig.angle).at[self.inner_index].add(inner.angle)
        return orig.__class__(angle=angle_outer + angle_inner, xy=xy_outer + xy_inner)


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

    def check_contacts(self, stated: StateDict) -> ContactWithMetadata:
        contacts = []
        for (n1, n2), fn in _CONTACT_FUNCTIONS.items():
            if stated[n1] is not None and stated[n2] is not None:
                contact, shape1, shape2 = fn(self.shaped, stated)
                len1, len2 = stated[n1].p.batch_size(), stated[n2].p.batch_size()
                offset1, offset2 = stated.offset(n1), stated.offset(n2)
                if n1 == n2:
                    outer_index, inner_index = generate_self_pairs(jnp.arange(len1))
                else:
                    outer_index, inner_index = generate_pairs(
                        jnp.arange(len1),
                        jnp.arange(len2),
                    )
                contact_with_meta = ContactWithMetadata(
                    contact=contact,
                    shape1=shape1.to_shape(),
                    shape2=shape2.to_shape(),
                    outer_index=outer_index + offset1,
                    inner_index=inner_index + offset2,
                )
                contacts.append(contact_with_meta)
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
    # Damping: dv/dt + vc = 0 -> v(t) = v0 * exp(-tc)
    # v(t + dt) = v0 * exp(-tc - dtc) = v0 * exp(-tc) * exp(-dtc) = v(t)exp(-dtc)
    # Thus, linear/angular damping factors are actually exp(-dtc)
    return state.replace(
        v=Velocity(angle=v_ang * space.angular_damping, xy=v_xy * space.linear_damping),
        f=state.f.zeros_like(),
    )


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
    dv = _dv2from1(v1, r1, v2, r2)
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
    v1 = solver.v1 - Velocity(
        angle=helper.inv_moment1 * jnp.cross(helper.r1, p),
        xy=p * helper.inv_mass1,
    )
    v2 = solver.v2 + Velocity(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, p),
        xy=p * helper.inv_mass2,
    )
    return solver.replace(v1=v1, v2=v2)


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
    dv = _dv2from1(solver.v1, helper.r1, solver.v2, helper.r2)
    vt = jnp.dot(dv, helper.tangent)
    dpt = -helper.mass_tangent * vt
    # Clamp friction impulse
    max_pt = contact.friction * solver.pn
    pt = jnp.clip(solver.pt + dpt, a_min=-max_pt, a_max=max_pt)
    dpt_clamped = helper.tangent * (pt - solver.pt)
    # Velocity update by contact tangent
    dvt1 = Velocity(
        angle=-helper.inv_moment1 * jnp.cross(helper.r1, dpt_clamped),
        xy=-dpt_clamped * helper.inv_mass1,
    )
    dvt2 = Velocity(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, dpt_clamped),
        xy=dpt_clamped * helper.inv_mass2,
    )
    # Compute Relative velocity again
    dv = _dv2from1(solver.v1 + dvt1, helper.r1, solver.v2 + dvt2, helper.r2)
    vn = _vmap_dot(dv, contact.normal)
    dpn = helper.mass_normal * (-vn + helper.v_bias)
    # Accumulate and clamp impulse
    pn = jnp.clip(solver.pn + dpn, a_min=0.0)
    dpn_clamped = contact.normal * (pn - solver.pn)
    # Velocity update by contact normal
    dvn1 = Velocity(
        angle=-helper.inv_moment1 * jnp.cross(helper.r1, dpn_clamped),
        xy=-dpn_clamped * helper.inv_mass1,
    )
    dvn2 = Velocity(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, dpn_clamped),
        xy=dpn_clamped * helper.inv_mass2,
    )
    # Filter dv
    dv1, dv2 = jax.tree_map(
        lambda x: jnp.where(solver.contact, x, jnp.zeros_like(x)),
        (dvn1 + dvt1, dvn2 + dvt2),
    )
    # Summing up dv per each contact pair
    return VelocitySolver(
        v1=dv1,
        v2=dv2,
        pn=pn,
        pt=pt,
        contact=solver.contact,
    )


@jax.vmap
def apply_bounce(
    contact: Contact,
    helper: ContactHelper,
    solver: VelocitySolver,
) -> tuple[Velocity, Velocity]:
    """
    Apply bounce (resititution).
    Suppose that each shape has (N_contact, 1) or (N_contact, 2).
    """
    # Relative veclocity (from shape2 to shape1)
    dv = _dv2from1(solver.v1, helper.r1, solver.v2, helper.r2)
    vn = jnp.dot(dv, contact.normal)
    pn = -helper.mass_normal * (vn + helper.bounce)
    dpn = contact.normal * pn
    # Velocity update by contact normal
    dv1 = Velocity(
        angle=-helper.inv_moment1 * jnp.cross(helper.r1, dpn),
        xy=-dpn * helper.inv_mass1,
    )
    dv2 = Velocity(
        angle=helper.inv_moment2 * jnp.cross(helper.r2, dpn),
        xy=dpn * helper.inv_mass2,
    )
    # Filter dv
    allow_bounce = jnp.logical_and(solver.contact, helper.allow_bounce)
    return jax.tree_map(
        lambda x: jnp.where(allow_bounce, x, jnp.zeros_like(x)),
        (dv1, dv2),
    )


@chex.dataclass
class PositionSolver:
    p1: Position
    p2: Position
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
    r1 = solver.p1.rotate(helper.local_anchor1)
    r2 = solver.p2.rotate(helper.local_anchor2)
    ga2_ga1 = r2 - r1 + solver.p2.xy - solver.p1.xy
    separation = jnp.dot(ga2_ga1, contact.normal) - contact.penetration
    c = jnp.clip(
        bias_factor * (separation + linear_slop),
        a_min=-max_linear_correction,
        a_max=0.0,
    )
    kn1 = _effective_mass(helper.inv_mass1, helper.inv_moment1, r1, contact.normal)
    kn2 = _effective_mass(helper.inv_mass2, helper.inv_moment2, r2, contact.normal)
    k_normal = kn1 + kn2
    impulse = jnp.where(k_normal > 0.0, -c / k_normal, jnp.zeros_like(c))
    pn = impulse * contact.normal
    p1 = Position(
        angle=-helper.inv_moment1 * jnp.cross(r1, pn),
        xy=-pn * helper.inv_mass1,
    )
    p2 = Position(
        angle=helper.inv_moment2 * jnp.cross(r2, pn),
        xy=pn * helper.inv_mass2,
    )
    min_sep = jnp.fmin(solver.min_separation, separation)
    # Filter separation
    p1, p2 = jax.tree_map(
        lambda x: jnp.where(solver.contact, x, jnp.zeros_like(x)),
        (p1, p2),
    )
    return solver.replace(p1=p1, p2=p2, min_separation=min_sep)


def solve_constraints(
    space: Space,
    solver: VelocitySolver,
    p: Position,
    v: Velocity,
    contact_with_meta: ContactWithMetadata,
) -> tuple[Velocity, Position, VelocitySolver]:
    """Resolve collisions by Sequential Impulse method"""
    outer, inner = contact_with_meta.outer_index, contact_with_meta.inner_index

    def get_pairs(p_or_v: _PositionLike) -> tuple[_PositionLike, _PositionLike]:
        return p_or_v.get_slice(outer), p_or_v.get_slice(inner)

    p1, p2 = get_pairs(p)
    v1, v2 = get_pairs(v)
    helper = init_contact_helper(
        space,
        contact_with_meta.contact,
        contact_with_meta.shape1,
        contact_with_meta.shape2,
        p1,
        p2,
        v1,
        v2,
    )
    # Warm up the velocity solver
    solver = apply_initial_impulse(
        contact_with_meta.contact,
        helper,
        solver.replace(v1=v1, v2=v2),
    )

    def vstep(
        _n_iter: int,
        vs: tuple[Velocity, VelocitySolver],
    ) -> tuple[Velocity, VelocitySolver]:
        v_i, solver_i = vs
        solver_i1 = apply_velocity_normal(contact_with_meta.contact, helper, solver_i)
        v_i1 = contact_with_meta.gather_p_or_v(solver_i1.v1, solver_i1.v2, v_i) + v_i
        v1, v2 = get_pairs(v_i1)
        return v_i1, solver_i1.replace(v1=v1, v2=v2)

    v, solver = jax.lax.fori_loop(0, space.n_velocity_iter, vstep, (v, solver))
    bv1, bv2 = apply_bounce(contact_with_meta.contact, helper, solver)
    v = contact_with_meta.gather_p_or_v(bv1, bv2, v) + v

    def pstep(
        _n_iter: int,
        ps: tuple[Position, PositionSolver],
    ) -> tuple[Position, PositionSolver]:
        p_i, solver_i = ps
        solver_i1 = correct_position(
            space.bias_factor,
            space.linear_slop,
            space.max_linear_correction,
            contact_with_meta.contact,
            helper,
            solver_i,
        )
        p_i1 = contact_with_meta.gather_p_or_v(solver_i1.p1, solver_i1.p2, p_i) + p_i
        p1, p2 = get_pairs(p_i1)
        return p_i1, solver_i1.replace(p1=p1, p2=p2)

    pos_solver = PositionSolver(
        p1=p1,
        p2=p2,
        contact=solver.contact,
        min_separation=jnp.zeros_like(p1.angle),
    )
    p, pos_solver = jax.lax.fori_loop(0, space.n_position_iter, pstep, (p, pos_solver))
    return v, p, solver


def dont_solve_constraints(
    _space: Space,
    solver: VelocitySolver,
    p: Position,
    v: Velocity,
    _contact_with_meta: ContactWithMetadata,
) -> tuple[Velocity, Position, VelocitySolver]:
    return v, p, solver


def step(space: Space, stated: StateDict, solver: VelocitySolver) -> StateDict:
    state = update_velocity(space, space.shaped.concat(), stated.concat())
    contact_with_meta = space.check_contacts(stated.update(state))
    # Check there's any penetration
    contacts = contact_with_meta.contact.penetration >= 0
    v, p, solver = jax.lax.cond(
        jnp.any(contacts),
        solve_constraints,
        dont_solve_constraints,
        space,
        solver.update(contacts),
        state.p,
        state.v,
        contact_with_meta,
    )
    statec = update_position(space, state.replace(v=v, p=p))
    return stated.update(statec)


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
    s = p1 - state.p.xy
    d, length = normalize(p2 - p1)
    t = -jnp.dot(s, d)
    c = s + t * d
    cc = jnp.linalg.norm(c)
    rr = (radius + circle.radius) ** 2
    fraction = t - jnp.sqrt(rr - cc)
    hitpoint = s + fraction * d
    normal, _ = normalize(hitpoint)
    return Raycast(  # type: ignore
        fraction=fraction / length,
        normal=normal,
        hit=jnp.logical_and(
            cc <= rr,
            jnp.logical_and(
                fraction >= 0.0,
                max_fraction * length >= fraction,
            ),
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
    numerator = jnp.dot(normal, v1 - p1)
    denominator = jnp.dot(normal, d)
    t = numerator / denominator
    p = p1 + t * d
    s = jnp.dot(p - v1, eunit)
    normal = jnp.where(numerator > 0.0, -normal, normal)
    return Raycast(  # type: ignore
        fraction=t,
        normal=normal,
        hit=jnp.logical_and(
            denominator != 0.0,
            jnp.logical_and(
                jnp.logical_and(t >= 0.0, max_fraction * length >= t),
                jnp.logical_and(s >= 0.0, length >= s),
            ),
        ),
    )
