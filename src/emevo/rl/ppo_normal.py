from __future__ import annotations

from typing import NamedTuple

import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.nn.initializers import orthogonal


def make_inormal(mean: jax.Array, logstd: jax.Array) -> distrax.Distribution:
    normal = distrax.LogStddevNormal(loc=mean, log_scale=logstd)
    return distrax.Independent(normal, reinterpreted_batch_ndims=1)


class Output(NamedTuple):
    mean: jax.Array
    logstd: jax.Array
    value: jax.Array

    def policy(self) -> distrax.Distribution:
        return make_inormal(self.mean, self.logstd)


class NormalPPONet(eqx.Module):
    torso: list[eqx.Module]
    value_head: eqx.nn.Linear
    mean_head: eqx.nn.Linear
    logstd_param: eqx.nn.Linear

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        action_size: int,
        key: jax.Array,
    ) -> None:
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        # Common layers
        self.torso = [
            eqx.nn.Linear(input_size, hidden_size, key=key1),
            jnp.tanh,
            eqx.nn.Linear(hidden_size, hidden_size, key=key2),
            jnp.tanh,
        ]
        self.value_head = eqx.nn.Linear(hidden_size, 1, key=key3)
        policy_head = eqx.nn.Linear(hidden_size, action_size, key=key4)
        # Use small value for policy initialization
        self.mean_head = eqx.tree_at(
            lambda linear: linear.weight,
            policy_head,
            orthogonal(scale=0.01)(key5, policy_head.weight.shape),
        )
        self.logstd_param = jnp.zeros((action_size,))

    def __call__(self, x: jax.Array) -> Output:
        for layer in self.torso:
            x = layer(x)
        value = self.value_head(x)
        mean = self.mean_head(x)
        logstd = jnp.ones_like(mean) * self.logstd_param
        return Output(mean=mean, logstd=logstd, value=value)

    def value(self, x: jax.Array) -> jax.Array:
        for layer in self.torso:
            x = layer(x)
        return self.value(x)


@chex.dataclass
class Rollout:
    """Rollout buffer that stores the entire history of one rollout"""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    values: jax.Array
    means: jax.Array
    logstds: jax.Array


@chex.dataclass(frozen=True, mappable_dataclass=False)
class Batch:
    """Batch for PPO, indexable to get a minibatch."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    advantages: jax.Array
    value_targets: jax.Array
    log_action_probs: jax.Array

    def __getitem__(self, idx: jax.Array):
        return self.__class__(  # type: ignore
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            advantages=self.advantages[idx],
            value_targets=self.value_targets[idx],
            log_action_probs=self.log_action_probs[idx],
        )


def compute_gae(
    r_t: jax.Array,
    discount_t: jax.Array,
    values: jax.Array,
    lambda_: float = 0.95,
) -> jax.Array:
    """Efficiently compute generalized advantage estimator (GAE)"""

    gamma_lambda_t = discount_t * lambda_
    delta_t = r_t + discount_t * values[1:] - values[:-1]
    n = delta_t.shape[0]

    def update(i: int, advantage_t: jax.Array) -> jax.Array:
        t = n - i - 1
        adv_t = delta_t[t] + gamma_lambda_t[t] * advantage_t[t + 1]
        return advantage_t.at[t].set(adv_t)

    advantage_t = jax.lax.fori_loop(0, n, update, jnp.zeros_like(values))
    return advantage_t[:-1]


def make_batch(
    rollout: Rollout,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> Batch:
    all_values = jnp.concatenate([rollout.values, next_value.reshape(1, -1)], axis=0)
    advantages = compute_gae(
        rollout.rewards,
        # Set γ = 0 when the episode terminates
        (1.0 - rollout.terminations) * gamma,
        all_values,
        gae_lambda,
    )
    value_targets = advantages + all_values[:-1]
    actions = rollout.actions
    log_action_probs = make_inormal(rollout.means, rollout.logstds).log_prob(actions)
    return Batch(
        observations=rollout.observations,
        actions=actions,
        # Convert (N, 1) shape to (N,)
        rewards=rollout.rewards.ravel(),
        advantages=advantages.ravel(),
        value_targets=value_targets.ravel(),
        log_action_probs=log_action_probs,
    )


def loss_function(
    network: NormalPPONet,
    batch: Batch,
    ppo_clip_eps: float,
    entropy_weight: float,
) -> jax.Array:
    net_out = jax.vmap(network)(batch.observations)
    # Policy loss
    policy_dist = make_inormal(net_out.mean, net_out.logstd)
    log_prob = policy_dist.log_prob(batch.actions)
    policy_ratio = jnp.exp(log_prob - batch.log_action_probs)
    clipped_ratio = jnp.clip(policy_ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
    clipped_objective = jnp.fmin(
        policy_ratio * batch.advantages,
        clipped_ratio * batch.advantages,
    )
    policy_loss = -jnp.mean(clipped_objective)
    # Value loss
    value_loss = jnp.mean(0.5 * (net_out.value - batch.value_targets) ** 2)
    # Entropy regularization
    entropy = jnp.mean(policy_dist.entropy())
    return policy_loss + value_loss - entropy_weight * entropy


vmapped_permutation = jax.vmap(jax.random.permutation, in_axes=(0, None), out_axes=0)


def get_minibatches(
    batch: Batch,
    key: chex.PRNGKey,
    minibatch_size: int,
    n_epochs: int,
) -> Batch:
    batch_size = batch.observations.shape[0]
    permutations = vmapped_permutation(jax.random.split(key, n_epochs), batch_size)

    def get_minibatch_impl(x: jax.Array) -> jax.Array:
        orig_shape = x.shape
        x = x[permutations]
        if len(orig_shape) == 1:
            return x.reshape(-1, minibatch_size)
        else:
            return x.reshape(-1, minibatch_size, *orig_shape[1:])

    return jax.tree_map(get_minibatch_impl, batch)


def update_network(
    batch: Batch,
    network: NormalPPONet,
    optax_update: optax.TransformUpdateFn,
    opt_state: optax.OptState,
    key: chex.PRNGKey,
    minibatch_size: int,
    n_epochs: int,
    ppo_clip_eps: float,
    entropy_weight: float,
) -> tuple[optax.OptState, NormalPPONet]:
    # Prepare update function
    dynamic_net, static_net = eqx.partition(network, eqx.is_array)

    def update_once(
        carried: tuple[optax.OptState, NormalPPONet],
        batch: Batch,
    ) -> tuple[tuple[optax.OptState, NormalPPONet], None]:
        opt_state, dynamic_net = carried
        network = eqx.combine(dynamic_net, static_net)
        grad = eqx.filter_grad(loss_function)(
            network,
            batch,
            ppo_clip_eps,
            entropy_weight,
        )
        updates, new_opt_state = optax_update(grad, opt_state)
        dynamic_net = optax.apply_updates(dynamic_net, updates)
        return (new_opt_state, dynamic_net), None

    # Prepare minibatches
    minibatches = get_minibatches(batch, key, minibatch_size, n_epochs)
    # Update network n_epochs x n_minibatches times
    (opt_state, updated_dynet), _ = jax.lax.scan(
        update_once,
        (opt_state, dynamic_net),
        minibatches,
    )
    return opt_state, eqx.combine(updated_dynet, static_net)
