"""Example of using circle foraging environment"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from emevo import genetic_ops as gops

Self = Any


class RewardFn(abc.ABC, eqx.Module):
    @abc.abstractmethod
    def serialise(self) -> dict[str, float | NDArray]:
        pass

    @abc.abstractmethod
    def __call__(self, *args) -> jax.Array:
        pass


RF = TypeVar("RF", bound=RewardFn)


def _item_or_np(array: jax.Array) -> float | NDArray:
    if array.ndim == 0:
        return array.item()
    else:
        return np.array(array)


def slice_last(w: jax.Array, i: int) -> jax.Array:
    return jnp.squeeze(jax.lax.slice_in_dim(w, i, i + 1, axis=-1))


def serialize_weight(w: jax.Array, keys: list[str]) -> dict[str, jax.Array]:
    return {key: slice_last(w, i) for i, key in enumerate(keys)}


class LinearReward(RewardFn):
    weight: jax.Array
    extractor: Callable[..., jax.Array]
    serializer: Callable[[jax.Array], dict[str, jax.Array]]

    def __init__(
        self,
        *,  # order of arguments are a bit confusing here...
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., jax.Array],
        serializer: Callable[[jax.Array], dict[str, jax.Array]],
        std: float = 1.0,
        mean: float = 0.0,
    ) -> None:
        self.weight = jax.random.normal(key, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        return jax.vmap(jnp.dot)(extracted, self.weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(_item_or_np, self.serializer(self.weight))


class SinhReward(RewardFn):
    weight: jax.Array
    extractor: Callable[..., jax.Array]
    serializer: Callable[[jax.Array], dict[str, jax.Array]]
    scale: float

    def __init__(
        self,
        *,  # order of arguments are a bit confusing here...
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., jax.Array],
        serializer: Callable[[jax.Array], dict[str, jax.Array]],
        scale: float = 2.5,
        std: float = 1.0,
        mean: float = 0.0,
    ) -> None:
        self.weight = jax.random.normal(key, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer
        self.scale = scale

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        return jax.vmap(jnp.dot)(extracted, jnp.sinh(self.weight * self.scale))

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(_item_or_np, self.serializer(self.weight))


class ExponentialReward(RewardFn):
    weight: jax.Array
    scale: jax.Array
    extractor: Callable[..., jax.Array]
    serializer: Callable[[jax.Array, jax.Array], dict[str, jax.Array]]

    def __init__(
        self,
        *,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., jax.Array],
        serializer: Callable[[jax.Array, jax.Array], dict[str, jax.Array]],
        std: float = 1.0,
        mean: float = 0.0,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (n_agents, n_weights)) * std + mean
        self.scale = jax.random.normal(k2, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        weight = (10**self.scale) * self.weight
        return jax.vmap(jnp.dot)(extracted, weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(
            _item_or_np,
            self.serializer(self.weight, self.scale),
        )


class BoundedExponentialReward(ExponentialReward):
    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        scale = (self.scale + 1.0) * 0.5
        weight = (10**scale) * self.weight
        return jax.vmap(jnp.dot)(extracted, weight)


class SigmoidReward(RewardFn):
    weight: jax.Array
    alpha: jax.Array
    extractor: Callable[..., tuple[jax.Array, jax.Array]]
    serializer: Callable[[jax.Array, jax.Array], dict[str, jax.Array]]

    def __init__(
        self,
        *,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., tuple[jax.Array, jax.Array]],
        serializer: Callable[[jax.Array, jax.Array], dict[str, jax.Array]],
        std: float = 1.0,
        mean: float = 0.0,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (n_agents, n_weights)) * std + mean
        self.alpha = jax.random.normal(k2, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer

    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        energy_alpha = energy.reshape(-1, 1) * self.alpha  # (N, n_weights)
        filtered = extracted / (1.0 + jnp.exp(-energy_alpha))
        return jax.vmap(jnp.dot)(filtered, self.weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(
            _item_or_np,
            self.serializer(self.weight, self.alpha),
        )


class SigmoidReward_01(SigmoidReward):
    """Scaled to [0, 1] for all alpha in [-1, 1]"""

    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        e = energy.reshape(-1, 1)  # (N, n_weights)
        alpha_plus = 2.0 * extracted / (1.0 + jnp.exp(-e * (1.0 - self.alpha))) - 1.0
        alpha_minus = 2.0 * extracted / (1.0 + jnp.exp(-e * self.alpha))
        filtered = jnp.where(self.alpha > 0, alpha_plus, alpha_minus)
        return jax.vmap(jnp.dot)(filtered, self.weight)


class SigmoidExponentialReward(RewardFn):
    weight: jax.Array
    scale: jax.Array
    alpha: jax.Array
    extractor: Callable[..., tuple[jax.Array, jax.Array]]
    serializer: Callable[[jax.Array, jax.Array, jax.Array], dict[str, jax.Array]]

    def __init__(
        self,
        *,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., tuple[jax.Array, jax.Array]],
        serializer: Callable[[jax.Array, jax.Array, jax.Array], dict[str, jax.Array]],
        std: float = 1.0,
        mean: float = 0.0,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.weight = jax.random.normal(k1, (n_agents, n_weights)) * std + mean
        self.scale = jax.random.normal(k2, (n_agents, n_weights)) * std + mean
        self.alpha = jax.random.normal(k3, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer

    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        weight = (10**self.scale) * self.weight
        e = energy.reshape(-1, 1)  # (N, n_weights)
        alpha_plus = 2.0 * extracted / (1.0 + jnp.exp(-e * (1.0 - self.alpha))) - 1.0
        alpha_minus = 2.0 * extracted / (1.0 + jnp.exp(-e * self.alpha))
        filtered = jnp.where(self.alpha > 0, alpha_plus, alpha_minus)
        return jax.vmap(jnp.dot)(filtered, weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(
            _item_or_np,
            self.serializer(self.weight, self.scale, self.alpha),
        )


class DelayedSEReward(RewardFn):
    weight: jax.Array
    scale: jax.Array
    delay: jax.Array
    extractor: Callable[..., tuple[jax.Array, jax.Array]]
    serializer: Callable[[jax.Array, jax.Array, jax.Array], dict[str, jax.Array]]
    delay_scale: float
    delay_neg_offset: float

    def __init__(
        self,
        *,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., tuple[jax.Array, jax.Array]],
        serializer: Callable[[jax.Array, jax.Array, jax.Array], dict[str, jax.Array]],
        std: float = 1.0,
        mean: float = 0.0,
        delay_scale: float = 20.0,
        delay_neg_offset: float = 20.0,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.weight = jax.random.normal(k1, (n_agents, n_weights)) * std + mean
        self.scale = jax.random.normal(k2, (n_agents, n_weights)) * std + mean
        self.delay = jax.random.normal(k3, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer
        self.delay_scale = delay_scale
        self.delay_neg_offset = delay_neg_offset

    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        weight = (10**self.scale) * self.weight
        e = energy.reshape(-1, 1)  # (N, n_weights)
        exp = jnp.exp(
            e * (self.delay < 0) - e * (self.delay > 0) + self.delay_scale * self.delay
        )
        filtered = extracted / (1.0 + exp)
        return jax.vmap(jnp.dot)(filtered, weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(
            _item_or_np,
            self.serializer(self.weight, self.scale, self.delay),
        )


class OffsetDelayedSEReward(DelayedSEReward):
    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        weight = (10**self.scale) * self.weight
        e = energy.reshape(-1, 1)  # (N, n_weights)
        exp_pos = jnp.exp(-e + self.delay_scale * self.delay)
        exp_neg = jnp.exp(
            e - self.delay_scale * (1.0 + self.delay) - self.delay_neg_offset
        )
        exp = jnp.where(self.delay > 0, exp_pos, exp_neg)
        filtered = extracted / (1.0 + exp)
        return jax.vmap(jnp.dot)(filtered, weight)


class OffsetDelayedSBEReward(DelayedSEReward):
    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        scale = (self.scale + 1.0) * 0.5
        weight = (10**scale) * self.weight
        e = energy.reshape(-1, 1)  # (N, n_weights)
        exp_pos = jnp.exp(-e + self.delay_scale * self.delay)
        exp_neg = jnp.exp(
            e - self.delay_scale * (1.0 + self.delay) - self.delay_neg_offset
        )
        exp = jnp.where(self.delay > 0, exp_pos, exp_neg)
        filtered = extracted / (1.0 + exp)
        return jax.vmap(jnp.dot)(filtered, weight)


class OffsetDelayedSinhReward(RewardFn):
    weight: jax.Array
    delay: jax.Array
    extractor: Callable[..., tuple[jax.Array, jax.Array]]
    serializer: Callable[[jax.Array, jax.Array], dict[str, jax.Array]]
    delay_scale: float
    scale: float

    def __init__(
        self,
        *,  # order of arguments are a bit confusing here...
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., tuple[jax.Array, jax.Array]],
        serializer: Callable[[jax.Array, jax.Array], dict[str, jax.Array]],
        std: float = 1.0,
        mean: float = 0.0,
        scale: float = 2.5,
        delay_scale: float = 20.0,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (n_agents, n_weights)) * std + mean
        self.delay = jax.random.normal(k2, (n_agents, n_weights)) * std + mean
        self.extractor = extractor
        self.serializer = serializer
        self.scale = scale
        self.delay_scale = delay_scale

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        extracted, energy = self.extractor(*args)
        weight = jnp.sinh(self.weight * self.scale)
        e = energy.reshape(-1, 1)  # (N, n_weights)
        exp_pos = jnp.exp(-e + self.delay_scale * self.delay)
        exp_neg = jnp.exp(e - self.delay_scale * (1.0 + self.delay) - self.delay_scale)
        exp = jnp.where(self.delay > 0, exp_pos, exp_neg)
        filtered = extracted / (1.0 + exp)
        return jax.vmap(jnp.dot)(filtered, weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_util.tree_map(
            _item_or_np,
            self.serializer(self.weight, self.delay),
        )


def mutate_reward_fn(
    key: chex.PRNGKey,
    reward_fn_dict: dict[int, RF],
    old: RF,
    mutation: gops.Mutation,
    parents: jax.Array,
    unique_id: jax.Array,
    slots: jax.Array,
) -> RF:
    n = parents.shape[0]
    dynamic_net, static_net = eqx.partition(old, eqx.is_array)
    keys = jax.random.split(key, n)
    for key, parent, uid, slot in zip(keys, parents, unique_id, slots):
        parent_reward_fn = reward_fn_dict[parent.item()]
        mutated_dnet = mutation(key, eqx.partition(parent_reward_fn, eqx.is_array)[0])
        reward_fn_dict[uid.item()] = eqx.combine(mutated_dnet, static_net)
        dynamic_net = jax.tree_util.tree_map(
            lambda orig, mutated: orig.at[slot.item()].set(mutated),
            dynamic_net,
            mutated_dnet,
        )
    return eqx.combine(dynamic_net, static_net)
