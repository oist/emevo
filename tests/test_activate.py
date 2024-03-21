import typing

import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import make
from emevo.environments.circle_foraging import CFState, CircleForaging

N_MAX_AGENTS = 10
N_INIT_AGENTS = 5
ENERGY_SHARE_RATIO = 0.4
INIT_ENERGY = 20.0


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env(key: chex.PRNGKey) -> tuple[CircleForaging, CFState]:
    env = make(
        "CircleForaging-v0",
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=N_INIT_AGENTS,
        init_energy=INIT_ENERGY,
        energy_share_ratio=ENERGY_SHARE_RATIO,
    )
    state, _ = env.reset(key)
    return typing.cast(CircleForaging, env), state


def test_deactivate(key: chex.PRNGKey) -> None:
    expected = jnp.array(
        [True, True, True, True, True, False, False, False, False, False]
    )
    env, state = reset_env(key)
    chex.assert_trees_all_close(state.unique_id.is_active(), expected)
    state = env.deactivate(state, jnp.zeros_like(expected).at[2].set(True))
    expected = jnp.array(
        [True, True, False, True, True, False, False, False, False, False]
    )
    chex.assert_trees_all_close(state.unique_id.is_active(), expected)
    nowhere = jnp.zeros((1, 2))
    is_nowhere = jnp.all(state.physics.circle.p.xy == nowhere, axis=-1)
    chex.assert_trees_all_close(is_nowhere, jnp.logical_not(expected))


def test_activate(key: chex.PRNGKey) -> None:
    env, state = reset_env(key)
    init_energy = state.status.energy
    is_parent = jnp.zeros(N_MAX_AGENTS, dtype=bool).at[jnp.array([2, 4, 7])].set(True)
    state, parents = env.activate(state, is_parent)
    expected_active = jnp.array(
        [True, True, True, True, True, True, True, False, False, False]
    )
    chex.assert_trees_all_close(state.unique_id.is_active(), expected_active)
    expected_parents = jnp.array([-1, -1, -1, -1, -1, 3, 5, -1, -1, -1])
    chex.assert_trees_all_close(parents, expected_parents)
    nowhere = jnp.zeros((1, 2))
    is_nowhere = jnp.all(state.physics.circle.p.xy == nowhere, axis=-1)
    chex.assert_trees_all_close(is_nowhere, jnp.logical_not(expected_active))
    expected_energy = (
        init_energy.at[jnp.array([2, 4])]
        .set(INIT_ENERGY * (1.0 - ENERGY_SHARE_RATIO))
        .at[jnp.array([5, 6])]
        .set(INIT_ENERGY * ENERGY_SHARE_RATIO)
    )
    chex.assert_trees_all_close(state.status.energy, expected_energy)
