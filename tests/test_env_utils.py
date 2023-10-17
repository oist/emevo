import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.phyjax2d import normalize
from emevo.environments.utils.food_repr import ReprLoc
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
