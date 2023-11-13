import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from emevo.rl.ppo_normal import (
    Batch,
    NormalPPONet,
    Rollout,
    get_minibatches,
    make_batch,
    update_network,
    vmap_batch,
    vmap_net,
    vmap_update,
)

OBS_SIZE = 10
ACT_SIZE = 4
STEP_SIZE = 512
MINIBATCH_SIZE = 64
N_EPOCHS = 4
N_MINIBATCHES = (STEP_SIZE // MINIBATCH_SIZE) * N_EPOCHS


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def _rollout() -> Rollout:
    return Rollout(
        observations=jnp.zeros((STEP_SIZE, OBS_SIZE)),
        actions=jnp.zeros((STEP_SIZE, ACT_SIZE)),
        rewards=(jnp.arange(STEP_SIZE) % 3).astype(jnp.float32).reshape(-1, 1),
        terminations=jnp.zeros((STEP_SIZE, 1), dtype=bool),
        values=jnp.zeros((STEP_SIZE, 1)),
        means=jnp.zeros((STEP_SIZE, ACT_SIZE)),
        logstds=jnp.ones((STEP_SIZE, ACT_SIZE)),
    )


def test_make_batch() -> None:
    rollout = _rollout()
    batch = make_batch(rollout, jnp.zeros((1,)), 0.99, 0.95)
    chex.assert_shape(batch.observations, (STEP_SIZE, OBS_SIZE))
    chex.assert_shape(batch.actions, (STEP_SIZE, ACT_SIZE))
    chex.assert_shape(batch.log_action_probs, (STEP_SIZE,))
    chex.assert_shape(batch.rewards, (STEP_SIZE,))
    chex.assert_shape(batch.advantages, (STEP_SIZE,))
    chex.assert_shape(batch.value_targets, (STEP_SIZE,))


def test_minibatches(key: chex.PRNGKey) -> None:
    rollout = _rollout()
    batch = make_batch(rollout, jnp.zeros((1,)), 0.99, 0.95)
    minibatch = get_minibatches(batch, key, MINIBATCH_SIZE, N_EPOCHS)
    prefix = N_MINIBATCHES, MINIBATCH_SIZE
    chex.assert_shape(minibatch.observations, (*prefix, OBS_SIZE))
    chex.assert_shape(minibatch.actions, (*prefix, ACT_SIZE))
    chex.assert_shape(minibatch.log_action_probs, (*prefix,))
    chex.assert_shape(minibatch.rewards, (*prefix,))
    chex.assert_shape(minibatch.advantages, (*prefix,))
    chex.assert_shape(minibatch.value_targets, (*prefix,))


def test_update_network(key: chex.PRNGKey) -> None:
    rollout = _rollout()
    batch = make_batch(rollout, jnp.zeros((1,)), 0.99, 0.95)
    key1, key2 = jax.random.split(key, 2)
    pponet = NormalPPONet(OBS_SIZE, 5, ACT_SIZE, key1)
    adam_init, adam_update = optax.adam(1e-3)
    opt_state = adam_init(eqx.filter(pponet, eqx.is_array))
    _, updated = update_network(
        batch,
        pponet,
        adam_update,
        opt_state,
        key2,
        64,
        10,
        0.1,
        0.01,
    )
    before, _ = eqx.partition(pponet, eqx.is_array)
    after, _ = eqx.partition(updated, eqx.is_array)
    chex.assert_trees_all_equal_shapes(before, after)


def test_ensemble(key: chex.PRNGKey) -> None:
    n = 3
    rollouts = jax.tree_map(
        lambda *args: jnp.stack(args, axis=1),
        *[_rollout() for _ in range(n)],
    )
    batch = vmap_batch(rollouts, jnp.zeros((n,)), 0.99, 0.95)
    chex.assert_shape(batch.observations, (n, STEP_SIZE, OBS_SIZE))

    key, net_key = jax.random.split(key)
    pponet = vmap_net(OBS_SIZE, 5, ACT_SIZE, jax.random.split(net_key, n))
    out = eqx.filter_vmap(lambda net, obs: jax.vmap(net)(obs))(
        pponet,
        batch.observations,
    )
    chex.assert_shape(out.mean, (n, STEP_SIZE, ACT_SIZE))

    adam_init, adam_update = optax.adam(1e-3)
    opt_state = jax.vmap(adam_init)(eqx.filter(pponet, eqx.is_array))

    _, updated = vmap_update(
        batch,
        pponet,
        adam_update,
        opt_state,
        jax.random.split(key, n),
        64,
        10,
        0.1,
        0.01,
    )
    before, _ = eqx.partition(pponet, eqx.is_array)
    after, _ = eqx.partition(updated, eqx.is_array)
    chex.assert_trees_all_equal_shapes(before, after)
