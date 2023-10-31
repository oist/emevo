import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.reproduction import  ReprNum
from emevo.environments.locating import (
    CircleCoordinate,
    Locating,
    LocatingFn,
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


def test_loc_gaussian(key: chex.PRNGKey) -> None:
    loc_g, state = Locating.GAUSSIAN((3.0, 3.0), (1.0, 1.0))
    loc = jax.vmap(loc_g, in_axes=(0, None))(jax.random.split(key, 10), state)
    chex.assert_shape(loc, (10, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 3) ** 2 < 1.0 and (y_mean - 3) ** 2 < 1.0


def test_loc_uniform(key: chex.PRNGKey) -> None:
    loc_u, state = Locating.UNIFORM(CircleCoordinate((3.0, 3.0), 3.0))
    loc = jax.vmap(loc_u, in_axes=(0, None))(jax.random.split(key, 10), state)
    chex.assert_shape(loc, (10, 2))
    bigger_circle = CircleCoordinate((3.0, 3.0), 4.0)
    assert jnp.all(jax.vmap(bigger_circle.contains_circle)(loc, jnp.ones(10)))


def test_loc_gm(key: chex.PRNGKey) -> None:
    loc_gm, state = Locating.GAUSSIAN_MIXTURE(
        [0.3, 0.7],
        ((0.0, 0.0), (10.0, 10.0)),
        ((1.0, 1.0), (1.0, 1.0)),
    )
    loc = jax.vmap(loc_gm, in_axes=(0, None))(jax.random.split(key, 20), state)
    chex.assert_shape(loc, (20, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 7) ** 2 < 1.0 and (y_mean - 7) ** 2 < 1.0


def test_loc_periodic(key: chex.PRNGKey) -> None:
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    loc_p, state = Locating.PERIODIC(*points)
    for i in range(10):
        loc = loc_p(key, state)
        state = state.increment()
        print(loc)
        assert jnp.all(loc == jnp.array(points[i % 3]))


def test_loc_switching(key: chex.PRNGKey) -> None:
    loc_g, _ = Locating.GAUSSIAN((3.0, 3.0), (1.0, 1.0))
    loc_u, _ = Locating.UNIFORM(CircleCoordinate((3.0, 3.0), 3.0))
    loc_s, state = Locating.SWITCHING(10, loc_g, loc_u)
    loc = jax.vmap(loc_s)(
        jax.random.split(key, 10),
        jax.tree_map(lambda a: jnp.tile(a, (10,)), state),
    )
    chex.assert_shape(loc, (10, 2))
    x_mean = jnp.mean(loc[:, 0])
    y_mean = jnp.mean(loc[:, 1])
    assert (x_mean - 3) ** 2 < 1.0 and (y_mean - 3) ** 2 < 1.0

    loc = jax.vmap(loc_s)(
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
