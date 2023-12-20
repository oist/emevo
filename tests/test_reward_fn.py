import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import genetic_ops as gops
from emevo.reward_fn import LinearReward, mutate_reward_fn


@pytest.fixture
def reward_fn() -> chex.PRNGKey:
    def slice_last(w: jax.Array, i: int) -> jax.Array:
        return jnp.squeeze(jax.lax.slice_in_dim(w, i, i + 1, axis=-1))

    return LinearReward(
        jax.random.PRNGKey(43),
        10,
        3,
        lambda x: x,  # Nothing to do
        lambda w: {"a": slice_last(w, 0), "b": slice_last(w, 1), "c": slice_last(w, 2)},
    )


def test_reward_fn(reward_fn: LinearReward) -> None:
    inputs = jnp.zeros((10, 3))
    reward = reward_fn(inputs)
    chex.assert_shape(reward, (10,))


def test_serialize(reward_fn: LinearReward) -> None:
    logd = reward_fn.serialize()
    chex.assert_shape((logd["a"], logd["b"], logd["c"]), (10,))


def test_mutation(reward_fn: LinearReward) -> None:
    reward_fn_dict = {i + 1: reward_fn.get_slice(i) for i in range(5)}
    chex.assert_shape(tuple(map(lambda lr: lr.weight, reward_fn_dict.values())), (3,))
    mutation = gops.GaussianMutation(std_dev=1.0)
    parents = jnp.array([-1, -1, -1, -1, -1, 2, 4, -1, -1, -1])
    mutated = mutate_reward_fn(
        jax.random.PRNGKey(23),
        reward_fn_dict,
        reward_fn,
        mutation,
        parents=parents[5:7],
        unique_id=jnp.array([6, 7]),
        slots=jnp.array([5, 6]),
    )
    same = parents == -1
    chex.assert_trees_all_close(reward_fn.weight[same], mutated.weight[same])
    different = parents != -1
    difference = reward_fn.weight[different] - mutated.weight[different]
    assert jnp.linalg.norm(difference) > 1e-6
    assert len(reward_fn_dict) == 7
    for i in range(7):
        chex.assert_trees_all_close(mutated.weight[i], reward_fn_dict[i + 1].weight)
