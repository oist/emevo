import typing

import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import TimeStep, make
from emevo.environments.cf_with_smell import (
    CFSObs,
    CFSState,
    CircleForagingWithSmell,
    _compute_smell,
    _vmap_compute_smell,
)

N_MAX_AGENTS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env(
    key: chex.PRNGKey,
) -> tuple[CircleForagingWithSmell, CFSState, TimeStep[CFSObs]]:
    # 12   x x O
    # 9  O x
    # 6      O  (O: agent, x: food)
    #    3 6 9 12
    env = make(
        "CircleForaging-v1",
        env_shape="square",
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=3,
        agent_loc_fn=(
            "periodic",
            [30.0, 90.0],
            [90.0, 60.0],
            [120.0, 120.0],
        ),
        food_loc_fn=(
            "periodic",
            [60.0, 90.0],
            [60.0, 120.0],
            [90.0, 120.0],
        ),
        food_num_fn=("constant", 3),
        foodloc_interval=20,
        agent_radius=AGENT_RADIUS,
        food_radius=FOOD_RADIUS,
    )
    state, timestep = env.reset(key)
    return typing.cast(CircleForagingWithSmell, env), state, timestep


def reset_multifood_env(
    key: chex.PRNGKey,
) -> tuple[CircleForagingWithSmell, CFSState, TimeStep[CFSObs]]:
    # 12   2 2 O
    # 9  O 1
    # 6      O  (O: agent, 1/2/3: food)
    #    3 6 9 12
    env = make(
        "CircleForaging-v1",
        env_shape="square",
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=3,
        agent_loc_fn=(
            "periodic",
            [30.0, 90.0],
            [90.0, 60.0],
            [120.0, 120.0],
        ),
        n_food_sources=2,
        food_loc_fn=[
            ("periodic", [60.0, 90.0]),  # 0
            ("periodic", [60.0, 120.0], [90.0, 120.0]),  # 1
        ],
        food_num_fn=[
            ("constant", 1),
            ("constant", 2),
        ],
        foodloc_interval=20,
        agent_radius=AGENT_RADIUS,
        food_radius=FOOD_RADIUS,
    )
    state, timestep = env.reset(key)
    return typing.cast(CircleForagingWithSmell, env), state, timestep


def test_smell1(key: chex.PRNGKey) -> None:
    env, state, ts = reset_env(key)

    smell = _compute_smell(1, 0.01, state.physics.static_circle, jnp.zeros(2))
    chex.assert_trees_all_close(smell, jnp.array([0.8235769]))
    chex.assert_shape(state.smell, (N_MAX_AGENTS, 1))

    chex.assert_trees_all_close(
        state.smell[:3].ravel(),
        jnp.array([1.95746815744733, 1.8619878481494374, 1.7593938269724911]),
    )

    chex.assert_shape(ts.obs.smell_diff, (N_MAX_AGENTS, 1))
    chex.assert_trees_all_close(ts.obs.smell_diff, jnp.zeros((N_MAX_AGENTS, 1)))

    _, ts = env.step(state, jnp.zeros((N_MAX_AGENTS, 2)).at[:3, 1].set(20.0))

    chex.assert_shape(ts.obs.smell_diff, (N_MAX_AGENTS, 1))
    sd = ts.obs.smell_diff[:3, 0]
    assert (sd[0] > 0.0).item()  # Get closer
    assert (sd[1] > 0.0).item()
    assert (sd[2] < 0.0).item()


def test_smell2(key: chex.PRNGKey) -> None:
    env, state, ts = reset_multifood_env(key)

    smell = _compute_smell(2, 0.01, state.physics.static_circle, jnp.zeros(2))
    chex.assert_trees_all_close(
        smell,
        jnp.array([0.33903043982484377, 0.4845465481658833]),
    )
    chex.assert_shape(state.smell, (N_MAX_AGENTS, 2))

    chex.assert_trees_all_close(
        state.smell[:3],
        jnp.array(
            [
                [0.7288934141100246, 1.2285747433373053],
                [0.6972891342043375, 1.1646987139451],
                [0.48621213667943447, 1.2731816902930566],
            ]
        ),
    )
    chex.assert_shape(ts.obs.smell_diff, (N_MAX_AGENTS, 2))
    chex.assert_trees_all_close(ts.obs.smell_diff, jnp.zeros((N_MAX_AGENTS, 2)))

    _, ts = env.step(state, jnp.zeros((N_MAX_AGENTS, 2)).at[:3, 1].set(20.0))

    chex.assert_shape(ts.obs.smell_diff, (N_MAX_AGENTS, 2))
    sd1 = ts.obs.smell_diff[:3, 0]
    assert (sd1[0] < 0.0).item()  # Get closer
    assert (sd1[1] > 0.0).item()
    assert (sd1[2] < 0.0).item()

    sd2 = ts.obs.smell_diff[:3, 1]
    assert (sd2[0] > 0.0).item()  # Get closer
    assert (sd2[1] > 0.0).item()
    assert (sd2[2] < 0.0).item()
