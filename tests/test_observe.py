import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import Env, make
from emevo.environments.circle_foraging import CFState, _observe_closest

N_MAX_AGENTS = 20
N_MAX_FOODS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env(key: chex.PRNGKey) -> tuple[Env, CFState]:
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
        agent_radius=10,
        food_radius=4,
    )
    return env, env.reset(key)


def test_observe(key: chex.PRNGKey) -> None:
    env, state = reset_env(key)
    obs = _observe_closest(
        env._physics.shaped,
        jnp.array([40.0, 10.0]),
        jnp.array([40.0, 30.0]),
        state.physics,
    )
    chex.assert_trees_all_close(obs, jnp.ones(3) * -1)
    obs = _observe_closest(
        env._physics.shaped,
        jnp.array([40.0, 10.0]),
        jnp.array([40.0, 110.0]),
        state.physics,
    )
    chex.assert_trees_all_close(obs, jnp.array([0.6, -1.0, -1.0]))
    obs = _observe_closest(
        env._physics.shaped,
        jnp.array([60.0, 10.0]),
        jnp.array([60.0, 110.0]),
        state.physics,
    )
    chex.assert_trees_all_close(obs, jnp.array([-1.0, 0.54, -1.0]))
    obs = _observe_closest(
        env._physics.shaped,
        jnp.array([110.0, 60.0]),
        jnp.array([90.0, 60.0]),
        state.physics,
    )
    chex.assert_trees_all_close(obs, jnp.array([-1.0, 0.7, -1.0]))
    obs = _observe_closest(
        env._physics.shaped,
        jnp.array([130.0, 60.0]),
        jnp.array([230.0, 60.0]),
        state.physics,
    )
    chex.assert_trees_all_close(obs, jnp.array([-1.0, -1.0, 0.3]))
