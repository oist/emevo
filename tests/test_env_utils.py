import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.phyjax2d import normalize
from emevo.environments.utils.food_repr import ReprLoc
from emevo.environments.utils.locating import CircleCoordinate, InitLoc


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def test_circle_coordinate(key: chex.PRNGKey) -> None:
    center = 3.0, 3.0
    circle = CircleCoordinate(center, 3.0)
    assert circle.contains_circle(jnp.array([3.0, 2.0]), jnp.array(1.1)).item()
    arr = circle.uniform(jax.random.PRNGKey(10))
    _, dist = normalize(arr - jnp.array(center))
    assert dist.item() <= 3.0
    jax.vmap(circle.uniform)(jax.random.split(key, 10))
