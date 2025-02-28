import chex
import jax.numpy as jnp
from phyjax2d import Force, Position, State, Velocity

from emevo.environments.smell import _compute_smell, _vmap_compute_smell


def test_smell_front() -> None:
    center = jnp.array([3.0, 4.0])
    nose = center + 3 ** (2**0.5)
    xy = jnp.array([[3.0, 1.0], [-1.0, 6.0], [6.0, 8.0], [14.0, 0.0]])
    p = Position(angle=jnp.zeros(4), xy=xy)
    state = State(
        p=p,
        v=Velocity.zeros(4),
        f=Force.zeros(4),
        is_active=jnp.ones(4, dtype=bool),
        label=jnp.zeros(4, dtype=jnp.int32),
    )
    smell = _compute_smell(0.1, True, state, nose, center)
    chex.assert_shape(smell, (4,))
    assert smell[0] == 0.0, smell
    assert smell[1] == 0.0
    assert smell[2] > 0.0
    assert smell[3] > 0.0


def test_vmap_smell() -> None:
    center = jnp.array([[3.0, 4.0], [8.0, 0.0]])
    nose = jnp.array([[3.0 + 3 ** (2**0.5), 4.0 + 3 ** (2**0.5)], [12.0, 0.0]])
    xy = jnp.array([[3.0, 1.0], [-1.0, 6.0], [6.0, 8.0], [14.0, 0.0]])
    p = Position(angle=jnp.zeros(4), xy=xy)
    state = State(
        p=p,
        v=Velocity.zeros(4),
        f=Force.zeros(4),
        is_active=jnp.ones(4, dtype=bool),
        label=jnp.zeros(4, dtype=jnp.int32),
    )
    smell = _vmap_compute_smell(0.1, True, state, nose, center)
    chex.assert_shape(smell, (2, 4))
    assert smell[0][0] == 0.0, smell
    assert smell[0][1] == 0.0
    assert smell[0][2] > 0.0
    assert smell[0][3] > 0.0
    assert smell[1][0] == 0.0
    assert smell[1][1] == 0.0
    assert smell[1][2] == 0.0
    assert smell[1][3] > 0.0


def test_smell_all() -> None:
    center = jnp.array([3.0, 4.0])
    nose = center + 3 ** (2**0.5)
    xy = jnp.array([[3.0, 1.0], [-1.0, 6.0], [6.0, 8.0], [14.0, 0.0]])
    p = Position(angle=jnp.zeros(4), xy=xy)
    state = State(
        p=p,
        v=Velocity.zeros(4),
        f=Force.zeros(4),
        is_active=jnp.ones(4, dtype=bool),
        label=jnp.zeros(4, dtype=jnp.int32),
    )
    smell = _compute_smell(0.1, False, state, nose, center)
    chex.assert_shape(smell, (4,))
    chex.assert_trees_all_equal(
        smell,
        jnp.array([0.40410987, 0.40070075, 0.8289342, 0.34136537]),
    )
