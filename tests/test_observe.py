import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import Env, make
from emevo.environments.locating import CircleCoordinate, Locating
from emevo.environments.phyjax2d import Space, StateDict
from emevo.environments.placement import place

N_MAX_AGENTS = 20
N_MAX_FOODS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env() -> Env:
    env = make(
        "CircleForaging-v0",
        env_shape="square",
        n_max_agents=10,
        n_initial_agents=5,
        agent_loc_fn=(
            "periodic",
            [40.0, 60.0],
            [60.0, 90.0],
            [80.0, 60.0],
            [100.0, 90.0],
            [120.0, 60.0],
        ),
        food_loc_fn=(
            "periodic",
            [60.0, 60.0],
            [80.0, 90.0],
            [80.0, 120.0],
            [100.0, 60.0],
        ),
        food_num_fn=("constant", 4),
        foodloc_interval=20,
    )


def test_observe(key: chex.PRNGKey) -> None:
    n = N_MAX_AGENTS // 2
    keys = jax.random.split(key, n)
