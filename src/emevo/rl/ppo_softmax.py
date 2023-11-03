from __future__ import annotations

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax.nn.initializers import orthogonal


class PPONetOutput(NamedTuple):
    policy_logits: jax.Array
    value: jax.Array


class SoftmaxPPONet(eqx.Module):
    torso: list
    value_head: eqx.nn.Linear
    policy_head: eqx.nn.Linear

    def __init__(self, key: jax.Array) -> None:
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        # Common layers
        self.torso = [
            eqx.nn.Conv2d(3, 1, kernel_size=3, key=key1),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(64, 64, key=key2),
            jax.nn.relu,
        ]
        self.value_head = eqx.nn.Linear(64, 1, key=key3)
        policy_head = eqx.nn.Linear(64, 4, key=key4)
        # Use small value for policy initialization
        self.policy_head = eqx.tree_at(
            lambda linear: linear.weight,
            policy_head,
            orthogonal(scale=0.01)(key5, policy_head.weight.shape),
        )

    def __call__(self, x: jax.Array) -> PPONetOutput:
        for layer in self.torso:
            x = layer(x)
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        return PPONetOutput(policy_logits=policy_logits, value=value)

    def value(self, x: jax.Array) -> jax.Array:
        for layer in self.torso:
            x = layer(x)
        return self.value_head(x)

@chex.dataclass
class Rollout:
    """Rollout buffer that stores the entire history of one rollout"""

    observations: jax.Array
    actions: jax.Array
    action_masks: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    values: jax.Array
    policy_logits: jax.Array


def mask_logits(policy_logits: jax.Array, action_mask: jax.Array) -> jax.Array:
    return jax.lax.select(
        action_mask,
        policy_logits,
        jnp.ones_like(policy_logits) * -jnp.inf,
    )


vmapped_obs2i = jax.vmap(obs_to_image)


@eqx.filter_jit
def exec_rollout(
    initial_state: State,
    initial_obs: Observation,
    env: jumanji.Environment,
    network: SoftmaxPPONet,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Rollout, Observation, jax.Array]:
    def step_rollout(
        carried: tuple[State, Observation],
        key: jax.Array,
    ) -> tuple[tuple[State, jax.Array], Rollout]:
        state_t, obs_t = carried
        obs_image = vmapped_obs2i(obs_t)
        net_out = jax.vmap(network)(obs_image)
        masked_logits = mask_logits(net_out.policy_logits, obs_t.action_mask)
        actions = jax.random.categorical(key, masked_logits, axis=-1)
        state_t1, timestep = jax.vmap(env.step)(state_t, actions)
        rollout = Rollout(
            observations=obs_image,
            actions=actions,
            action_masks=obs_t.action_mask,
            rewards=timestep.reward,
            terminations=1.0 - timestep.discount,
            values=net_out.value,
            policy_logits=masked_logits,
        )
        return (state_t1, timestep.observation), rollout

    (state, obs), rollout = jax.lax.scan(
        step_rollout,
        (initial_state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    next_value = jax.vmap(network.value)(vmapped_obs2i(obs))
    return state, rollout, obs, next_value

@chex.dataclass(frozen=True, mappable_dataclass=False)
class Batch:
    """Batch for PPO, indexable to get a minibatch."""

    observations: jax.Array
    action_masks: jax.Array
    onehot_actions: jax.Array
    rewards: jax.Array
    advantages: jax.Array
    value_targets: jax.Array
    log_action_probs: jax.Array

    def __getitem__(self, idx: jax.Array):
        return self.__class__(  # type: ignore
            observations=self.observations[idx],
            action_masks=self.action_masks[idx],
            onehot_actions=self.onehot_actions[idx],
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


@eqx.filter_jit
def make_batch(
    rollout: Rollout,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> Batch:
    all_values = jnp.concatenate(
        [jnp.squeeze(rollout.values), next_value.reshape(1, -1)]
    )
    advantages = compute_gae(
        rollout.rewards,
        # Set γ = 0 when the episode terminates
        (1.0 - rollout.terminations) * gamma,
        all_values,
        gae_lambda,
    )
    value_targets = advantages + all_values[:-1]
    onehot_actions = jax.nn.one_hot(rollout.actions, 4)
    _, _, *obs_shape = rollout.observations.shape
    log_action_probs = jnp.sum(
        jax.nn.log_softmax(rollout.policy_logits) * onehot_actions,
        axis=-1,
    )
    return Batch(
        observations=rollout.observations.reshape(-1, *obs_shape),
        action_masks=rollout.action_masks.reshape(-1, 4),
        onehot_actions=onehot_actions.reshape(-1, 4),
        rewards=rollout.rewards.ravel(),
        advantages=advantages.ravel(),
        value_targets=value_targets.ravel(),
        log_action_probs=log_action_probs.ravel(),
    )


def loss_function(
    network: SoftmaxPPONet,
    batch: Batch,
    ppo_clip_eps: float,
) -> jax.Array:
    net_out = jax.vmap(network)(batch.observations)
    # Policy loss
    log_pi = jax.nn.log_softmax(
        jax.lax.select(
            batch.action_masks,
            net_out.policy_logits,
            jnp.ones_like(net_out.policy_logits * -jnp.inf),
        )
    )
    log_action_probs = jnp.sum(log_pi * batch.onehot_actions, axis=-1)
    policy_ratio = jnp.exp(log_action_probs - batch.log_action_probs)
    clipped_ratio = jnp.clip(policy_ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
    clipped_objective = jnp.fmin(
        policy_ratio * batch.advantages,
        clipped_ratio * batch.advantages,
    )
    policy_loss = -jnp.mean(clipped_objective)
    # Value loss
    value_loss = jnp.mean(0.5 * (net_out.value - batch.value_targets) ** 2)
    # Entropy regularization
    entropy = jnp.mean(-jnp.exp(log_pi) * log_pi)
    return policy_loss + value_loss - 0.01 * entropy


vmapped_permutation = jax.vmap(jax.random.permutation, in_axes=(0, None), out_axes=0)


@eqx.filter_jit
def update_network(
    batch: Batch,
    network: SoftmaxPPONet,
    optax_update: optax.TransformUpdateFn,
    opt_state: optax.OptState,
    prng_key: jax.Array,
    minibatch_size: int,
    n_epochs: int,
    ppo_clip_eps: float,
) -> tuple[optax.OptState, SoftmaxPPONet]:
    # Prepare update function
    dynamic_net, static_net = eqx.partition(network, eqx.is_array)

    def update_once(
        carried: tuple[optax.OptState, SoftmaxPPONet],
        batch: Batch,
    ) -> tuple[tuple[optax.OptState, SoftmaxPPONet], None]:
        opt_state, dynamic_net = carried
        network = eqx.combine(dynamic_net, static_net)
        grad = eqx.filter_grad(loss_function)(network, batch, ppo_clip_eps)
        updates, new_opt_state = optax_update(grad, opt_state)
        dynamic_net = optax.apply_updates(dynamic_net, updates)
        return (new_opt_state, dynamic_net), None

    # Prepare minibatches
    batch_size = batch.observations.shape[0]
    permutations = vmapped_permutation(jax.random.split(prng_key, n_epochs), batch_size)
    minibatches = jax.tree_map(
        # Here, x's shape is [batch_size, ...]
        lambda x: x[permutations].reshape(-1, minibatch_size, *x.shape[1:]),
        batch,
    )
    # Update network n_epochs x n_minibatches times
    (opt_state, updated_dynet), _ = jax.lax.scan(
        update_once,
        (opt_state, dynamic_net),
        minibatches,
    )
    return opt_state, eqx.combine(updated_dynet, static_net)
