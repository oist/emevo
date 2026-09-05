import typing
from collections.abc import Callable
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from emevo import make
from emevo.environments.circle_foraging import CFObs, CFState, CircleForaging

from emevo.exp_utils import (
    EvolutionSnapshot,
    LogMode,
    Logger,
    load_snapshot,
)

N_MAX_AGENTS = 10
N_INIT_AGENTS = 5
ENERGY_SHARE_RATIO = 0.4
INIT_ENERGY = 20.0


class DummyModule(eqx.Module):
    weight: jax.Array
    activation: Callable[[jax.Array], jax.Array] = jnp.tanh


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env(key: chex.PRNGKey) -> tuple[CircleForaging, CFState, CFObs]:
    env = make(
        "CircleForaging-v0",
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=N_INIT_AGENTS,
        init_energy=INIT_ENERGY,
        energy_share_ratio=ENERGY_SHARE_RATIO,
    )
    state, timestep = env.reset(key)
    return typing.cast(CircleForaging, env), state, timestep.obs


def test_snapshot_roundtrip_with_dummy_data(tmp_path: Path) -> None:
    logger = Logger(tmp_path, LogMode.NONE, 10, 10, 0)
    snapshot = EvolutionSnapshot(
        epoch=12,
        env_state={"position": jnp.array([1.0, 2.0])},
        obs=jnp.array([3.0]),
        opt_state={"count": jnp.array(5)},
        network=DummyModule(jnp.array([6.0])),
        reward_fn=DummyModule(jnp.array([7.0])),
        prng_key=jnp.array([8, 9], dtype=jnp.uint32),
    )

    path = logger.save_snapshot(snapshot)
    assert path == tmp_path / "snapshot-12.hdf5"
    restored = load_snapshot(path)

    assert restored.epoch == 12
    np.testing.assert_array_equal(restored.env_state["position"], [1.0, 2.0])
    assert isinstance(restored.network, DummyModule)
    assert isinstance(restored.reward_fn, DummyModule)
    np.testing.assert_array_equal(restored.network.weight, [6.0])
    np.testing.assert_array_equal(restored.reward_fn.weight, [7.0])
    np.testing.assert_array_equal(restored.network.activation(jnp.array([0.0])), [0.0])
    np.testing.assert_array_equal(restored.prng_key, [8, 9])


def test_logger_restore_state(tmp_path: Path) -> None:
    source = Logger(tmp_path, LogMode.NONE, 10, 10, 0)
    source.reward_fn_dict[1] = {"value": 2}  # type: ignore[assignment]
    source._log_index = 7

    restored = Logger(tmp_path, LogMode.NONE, 10, 10, 0)
    restored.restore_state(source.get_state())

    assert restored.reward_fn_dict == {1: {"value": 2}}
    assert restored._log_index == 7


def test_circle_foraging_snapshot_roundtrip(
    tmp_path: Path,
    key: chex.PRNGKey,
) -> None:
    _, env_state, obs = reset_env(key)
    logger = Logger(tmp_path, LogMode.NONE, 10, 10, 0)
    snapshot = EvolutionSnapshot(
        epoch=3,
        env_state=env_state,
        obs=obs,
        opt_state={"count": jnp.array(1)},
        network=DummyModule(jnp.array([2.0])),
        reward_fn=DummyModule(jnp.array([3.0])),
        prng_key=key,
    )

    restored = load_snapshot(logger.save_snapshot(snapshot))

    assert isinstance(restored.env_state, CFState)
    assert isinstance(restored.obs, CFObs)
    assert isinstance(restored.network, DummyModule)
    assert isinstance(restored.reward_fn, DummyModule)
    chex.assert_trees_all_close(restored.env_state, env_state)
    chex.assert_trees_all_close(restored.obs, obs)
