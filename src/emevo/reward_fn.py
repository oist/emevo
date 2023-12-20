"""Example of using circle foraging environment"""
from __future__ import annotations

import abc
from typing import Any, Callable

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
    def serialize(self) -> dict[str, float | NDArray]:
        pass

    def get_slice(
        self,
        slice_idx: int | jax.Array,
        include_static: bool = False,
    ) -> Self:
        dynamic, static = eqx.partition(self, eqx.is_array)
        sliced_dyn = jax.tree_map(lambda item: item[slice_idx], dynamic)
        if include_static:
            return eqx.combine(sliced_dyn, static)
        else:
            return sliced_dyn


def _item_or_np(array: jax.Array) -> float | NDArray:
    if array.ndim == 0:
        return array.item()
    else:
        return np.array(array)


class LinearReward(RewardFn):
    weight: jax.Array
    extractor: Callable[..., jax.Array]
    serializer: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        key: chex.PRNGKey,
        n_agents: int,
        n_weights: int,
        extractor: Callable[..., jax.Array],
        serializer: Callable[[jax.Array], dict[str, jax.Array]],
    ) -> None:
        self.weight = jax.random.normal(key, (n_agents, n_weights))
        self.extractor = extractor
        self.serializer = serializer

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        return jax.vmap(jnp.dot)(extracted, self.weight)

    def serialize(self) -> dict[str, float | NDArray]:
        return jax.tree_map(_item_or_np, self.serializer(self.weight))


def mutate_reward_fn(
    key: chex.PRNGKey,
    reward_fn_dict: dict[int, eqx.Module],
    old: eqx.Module,
    mutation: gops.Mutation,
    parents: jax.Array,
    unique_id: jax.Array,
) -> eqx.Module:
    # new[i] := old[i] if i not in parents
    # new[i] := mutation(old[parents[i]]) if i in parents
    is_parent = parents != -1
    if not jnp.any(is_parent):
        return old
    dynamic_net, static_net = eqx.partition(old, eqx.is_array)
    keys = jax.random.split(key, jnp.sum(is_parent).item())
    for i, key in zip(jnp.nonzero(is_parent)[0], keys):
        parent_reward_fn = reward_fn_dict[parents[i].item()]
        mutated_dnet = mutation(key, parent_reward_fn)
        reward_fn_dict[unique_id[i].item()] = eqx.combine(mutated_dnet, static_net)
        dynamic_net = jax.tree_map(
            lambda orig, mutated: orig.at[i].set(mutated),
            dynamic_net,
            mutated_dnet,
        )
    return eqx.combine(dynamic_net, static_net)
