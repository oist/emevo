import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import Env, make
from emevo.environments.circle_foraging import CFState, _observe_closest, get_sensor_obs

N_MAX_AGENTS = 20
N_MAX_FOODS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env(key: chex.PRNGKey) -> tuple[Env, CFState]:
    #     x
    #   O x O
    # O x O  O  (O: agent, x: food)
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
        ),
        food_num_fn=("constant", 3),
        foodloc_interval=20,
        agent_radius=10,
        food_radius=4,
    )
    return env, env.reset(key)


def test_observe_closest(key: chex.PRNGKey) -> None:
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
        jnp.array([130.0, 60.0]),
        jnp.array([230.0, 60.0]),
        state.physics,
    )
    chex.assert_trees_all_close(obs, jnp.array([-1.0, -1.0, 0.3]))


def test_sensor_obs(key: chex.PRNGKey) -> None:
    env, state = reset_env(key)
    sensor_obs = get_sensor_obs(
        env._physics.shaped,
        3,
        (-90, 90),
        100.0,
        state.physics,
    )
    chex.assert_shape(sensor_obs, (30, 3))
    # Food is to the right/left of the circle
    chex.assert_trees_all_close(
        sensor_obs[0],
        sensor_obs[3],
        sensor_obs[8],
        sensor_obs[11],
        jnp.array([-1.0, 0.94, -1.0]),
    )
    # Food is above the circle
    chex.assert_trees_all_close(sensor_obs[7], jnp.array([-1.0, 0.84, -1.0]))
    # They can see each other
    chex.assert_trees_all_close(
        sensor_obs[6],
        sensor_obs[14],
        jnp.array([0.8, -1.0, -1.0]),
    )
    # Walls
    chex.assert_trees_all_close(sensor_obs[2], jnp.array([-1.0, -1.0, 0.7]))
    chex.assert_trees_all_close(sensor_obs[5], jnp.array([-1.0, -1.0, 0.5]))
    chex.assert_trees_all_close(sensor_obs[9], jnp.array([-1.0, -1.0, 0.1]))
    chex.assert_trees_all_close(
        sensor_obs[4],
        sensor_obs[10],
        jnp.array([-1.0, -1.0, 0.0]),
    )
    chex.assert_trees_all_close(sensor_obs[12], jnp.array([-1.0, -1.0, 0.3]))
    # Nothing
    chex.assert_trees_all_close(
        sensor_obs[1],
        sensor_obs[13],
        jnp.array([-1.0, -1.0, -1.0]),
    )


def test_encount(key: chex.PRNGKey) -> None:
    env, state = reset_env(key)
