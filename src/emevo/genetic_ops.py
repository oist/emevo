""" Genetics operations for pytree"""

from __future__ import annotations

import abc
import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp


class Crossover(abc.ABC):
    @abc.abstractmethod
    def _select(
        self,
        prng_key: chex.PRNGKey,
        array1: jax.Array,
        array2: jax.Array,
    ) -> jax.Array:
        pass

    def __call__(
        self,
        prng_key: chex.PRNGKey,
        params_a: chex.ArrayTree,
        params_b: chex.ArrayTree,
    ) -> chex.ArrayTree:
        leaves, treedef = jax.tree_util.tree_flatten(params_a)
        prng_keys = jax.random.split(prng_key, len(leaves))
        result = jax.tree_map(
            self._select,
            treedef.unflatten(prng_keys),
            params_a,
            params_b,
        )
        return result


class Mutation(abc.ABC):
    @abc.abstractmethod
    def _add_noise(self, prng_key: chex.PRNGKey, array: jax.Array) -> jax.Array:
        pass

    def __call__(
        self,
        prng_key: chex.PRNGKey,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        leaves, treedef = jax.tree_util.tree_flatten(params)
        prng_keys = jax.random.split(prng_key, len(leaves))
        result = jax.tree_map(self._add_noise, treedef.unflatten(prng_keys), params)
        return result


@dataclasses.dataclass(frozen=True)
class UniformCrossover(Crossover):
    bias: float

    def __post_init__(self) -> None:
        assert self.bias >= 0.0 and self.bias <= 0.5

    def _select(
        self,
        prng_key: chex.PRNGKey,
        array1: jax.Array,
        array2: jax.Array,
    ) -> jax.Array:
        flags = jax.random.bernoulli(
            prng_key,
            p=self.bias + 0.5,
            shape=array1.shape,
        )
        return cast(jax.Array, jnp.where(flags, array1, array2))


@dataclasses.dataclass(frozen=True)
class CrossoverAndMutation(Crossover):
    crossover: Crossover
    mutation: Mutation

    def _select(
        self,
        prng_key: chex.PRNGKey,
        array1: jax.Array,
        array2: jax.Array,
    ) -> jax.Array:
        key1, key2 = jax.random.split(prng_key)
        selected = self.crossover._select(key1, array1, array2)
        return self.mutation._add_noise(key2, selected)


@dataclasses.dataclass(frozen=True)
class BernoulliMixtureMutation(Mutation):
    mutation_prob: float
    mutator: Mutation

    def _add_noise(self, prng_key: chex.PRNGKey, array: jax.Array) -> jax.Array:
        key1, key2 = jax.random.split(prng_key)
        noise_added = self.mutator._add_noise(key1, array)
        is_mutated = jax.random.bernoulli(
            key2,
            self.mutation_prob,
            shape=array.shape,
        )
        return cast(jax.Array, jnp.where(is_mutated, noise_added, array))


@dataclasses.dataclass(frozen=True)
class GaussianMutation(Mutation):
    std_dev: float

    def _add_noise(self, prng_key: chex.PRNGKey, array: jax.Array) -> jax.Array:
        std_normal = jax.random.normal(prng_key, shape=array.shape)
        return array + std_normal * self.std_dev


@dataclasses.dataclass(frozen=True)
class UniformMutation(Mutation):
    max_noise: float

    def _add_noise(self, prng_key: chex.PRNGKey, array: jax.Array) -> jax.Array:
        uniform = jax.random.uniform(
            prng_key,
            shape=array.shape,
            maxval=self.max_noise * 2,
        )
        return array + uniform - self.max_noise
