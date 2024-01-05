"""Example of using circle foraging environment"""
from __future__ import annotations

import abc
from typing import Any, Callable, TypeVar

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


class LinearReward(RewardFn):
    weight: jax.Array
    extractor: Callable[..., jax.Array]
    serialiser: Callable[[jax.Array], dict[str, jax.Array]]

    def __init__(
        self,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., jax.Array],
        serialiser: Callable[[jax.Array], dict[str, jax.Array]],
    ) -> None:
        self.weight = jax.random.normal(key, (n_agents, n_weights))
        self.extractor = extractor
        self.serialiser = serialiser

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        return jax.vmap(jnp.dot)(extracted, self.weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_map(_item_or_np, self.serialiser(self.weight))


class SigmoidReward(RewardFn):
    weight: jax.Array
    alpha: jax.Array
    extractor: Callable[..., tuple[jax.Array, jax.Array]]
    serialiser: Callable[[jax.Array, jax.Array], dict[str, jax.Array]]

    def __init__(
        self,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., tuple[jax.Array, jax.Array]],
        serialiser: Callable[[jax.Array, jax.Array], dict[str, jax.Array]],
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (n_agents, n_weights))
        self.alpha = jax.random.normal(k2, (n_agents, n_weights))
        self.extractor = extractor
        self.serialiser = serialiser

    def __call__(self, *args) -> jax.Array:
        extracted, energy = self.extractor(*args)
        energy_alpha = energy.reshape(-1, 1) * self.alpha  # (N, n_weights)
        filtered = extracted / (1.0 + jnp.exp(-energy_alpha))
        return jax.vmap(jnp.dot)(filtered, self.weight)

    def serialise(self) -> dict[str, float | NDArray]:
        return jax.tree_map(_item_or_np, self.serialiser(self.weight, self.alpha))


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
        dynamic_net = jax.tree_map(
            lambda orig, mutated: orig.at[slot.item()].set(mutated),
            dynamic_net,
            mutated_dnet,
        )
    return eqx.combine(dynamic_net, static_net)
