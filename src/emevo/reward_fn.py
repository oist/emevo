"""Example of using circle foraging environment"""
from __future__ import annotations

import abc
from typing import Callable, Protocol

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from emevo import genetic_ops as gops


class RewardFn(abc.ABC, eqx.Module):
    @abc.abstractmethod
    def as_logdict(self) -> dict[str, float | NDArray]:
        pass


class LinearReward(RewardFn):
    weight: jax.Array
    extractor: Callable[..., jax.Array]
    serializer: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        key: chex.PRNGKey,
        n_agents: int,
        extractor: Callable[..., jax.Array],
    ) -> None:
        self.weight = jax.random.normal(key, (n_agents, 4))
        self.extractor = extractor

    def __call__(self, *args) -> jax.Array:
        extracted = self.extractor(*args)
        return jax.vmap(jnp.dot)(extracted, self.weight)

    def as_logdict(self) -> dict[str, float | NDArray]:
        return {""}


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
        parent_reward_fn = reward_fn_dict[parents[i]]
        mutated_dnet = mutation(key, parent_reward_fn)
        reward_fn_dict[unique_id[i]] = eqx.combine(mutated_dnet, static_net)
        dynamic_net = jax.tree_map(lambda arr: arr[i].set(mutated_dnet), dynamic_net)
    return eqx.combine(dynamic_net, static_net)
