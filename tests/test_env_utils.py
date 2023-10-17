import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.utils.food_repr import ReprLoc, ReprNum
from emevo.environments.utils.locating import (
    CircleCoordinate,
    InitLoc,
    SquareCoordinate,
)


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def test_circle_coordinate(key: chex.PRNGKey) -> None:
    circle = CircleCoordinate((3.0, 3.0), 3.0)
    assert circle.contains_circle(jnp.array([3.0, 2.0]), jnp.array(1.0)).item()
    assert not circle.contains_circle(jnp.array([5.0, 0.0]), jnp.array(1.0)).item()
    arr = jax.vmap(circle.uniform)(jax.random.split(key, 10))
    chex.assert_shape(arr, (10, 2))
    bigger_circle = CircleCoordinate((3.0, 3.0), 4.0)
    assert jnp.all(jax.vmap(bigger_circle.contains_circle)(arr, jnp.ones(10)))


def test_square_coordinate(key: chex.PRNGKey) -> None:
    square = SquareCoordinate((-2.0, 2.0), (1.0, 4.0))
    assert square.contains_circle(jnp.array([0.0, 3.0]), jnp.array(1.0)).item()
    assert not square.contains_circle(jnp.array([0.0, 4.0]), jnp.array(1.0)).item()
    arr = jax.vmap(square.uniform)(jax.random.split(key, 10))
    chex.assert_shape(arr, (10, 2))
    bigger_square = SquareCoordinate((-3.0, 3.0), (0.0, 5.0))
    assert jnp.all(jax.vmap(bigger_square.contains_circle)(arr, jnp.ones(10))), arr


def test_initloc_gaussian(key: chex.PRNGKey) -> None:
    initloc_g = InitLoc.GAUSSIAN((3.0, 3.0), (1.0, 1.0))
    loc = jax.vmap(initloc_g)(jax.random.split(key, 10))
    chex.assert_shape(loc, (10, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 3) ** 2 < 1.0 and (y_mean - 3) ** 2 < 1.0


def test_initloc_uniform(key: chex.PRNGKey) -> None:
    initloc_u = InitLoc.UNIFORM(CircleCoordinate((3.0, 3.0), 3.0))
    loc = jax.vmap(initloc_u)(jax.random.split(key, 10))
    chex.assert_shape(loc, (10, 2))
    bigger_circle = CircleCoordinate((3.0, 3.0), 4.0)
    assert jnp.all(jax.vmap(bigger_circle.contains_circle)(loc, jnp.ones(10)))


def test_initloc_gm(key: chex.PRNGKey) -> None:
    initloc_gm = InitLoc.GAUSSIAN_MIXTURE(
        [0.3, 0.7],
        ((0.0, 0.0), (10.0, 10.0)),
        ((1.0, 1.0), (1.0, 1.0)),
    )
    loc = jax.vmap(initloc_gm)(jax.random.split(key, 20))
    chex.assert_shape(loc, (20, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 7) ** 2 < 1.0 and (y_mean - 7) ** 2 < 1.0


def test_initloc_choice(key: chex.PRNGKey) -> None:
    initloc_c = InitLoc.CHOICE([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    loc = jax.vmap(initloc_c)(jax.random.split(key, 20))
    chex.assert_shape(loc, (20, 2))
    c1 = loc == jnp.array([[0.0, 0.0]])
    c2 = loc == jnp.array([[1.0, 1.0]])
    c3 = loc == jnp.array([[2.0, 2.0]])
    assert jnp.all(jnp.logical_or(c1, jnp.logical_or(c2, c3)))


def test_reprloc_gaussian(key: chex.PRNGKey) -> None:
    reprloc_g, state = ReprLoc.GAUSSIAN((3.0, 3.0), (1.0, 1.0))
    loc = jax.vmap(reprloc_g)(
        jax.random.split(key, 10),
        jax.tree_map(lambda a: jnp.tile(a, (10,)), state),
    )
    chex.assert_shape(loc, (10, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 3) ** 2 < 1.0 and (y_mean - 3) ** 2 < 1.0


def test_reprloc_switching(key: chex.PRNGKey) -> None:
    initloc_g = InitLoc.GAUSSIAN((3.0, 3.0), (1.0, 1.0))
    initloc_u = InitLoc.UNIFORM(CircleCoordinate((3.0, 3.0), 3.0))
    reprloc_s, state = ReprLoc.SWITCHING(10, initloc_g, initloc_u)
    loc = jax.vmap(reprloc_s)(
        jax.random.split(key, 10),
        jax.tree_map(lambda a: jnp.tile(a, (10,)), state),
    )
    chex.assert_shape(loc, (10, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 3) ** 2 < 1.0 and (y_mean - 3) ** 2 < 1.0

    loc = jax.vmap(reprloc_s)(
        jax.random.split(key, 10),
        jax.tree_map(lambda a: jnp.tile(a * 10, (10,)), state),
    )
    chex.assert_shape(loc, (10, 2))
    bigger_circle = CircleCoordinate((3.0, 3.0), 4.0)
    assert jnp.all(jax.vmap(bigger_circle.contains_circle)(loc, jnp.ones(10)))


def test_foodnum_const() -> None:
    const, state = ReprNum.CONSTANT(10)
    assert const(state.eaten(3)).appears()
    assert const(state.eaten(3).recover(2)).appears()
    assert not const(state.eaten(3).recover(3)).appears()
