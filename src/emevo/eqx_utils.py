from __future__ import annotations

from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

M = TypeVar("M", bound=eqx.Module)


def get_slice(module: M, slice_idx: int | jax.Array) -> M:
    dynamic, static = eqx.partition(module, eqx.is_array)
    sliced_dyn = jax.tree_map(lambda item: item[slice_idx], dynamic)
    return eqx.combine(sliced_dyn, static)


@eqx.filter_jit
def where(flag: jax.Array, mod_a: M, mod_b: M) -> M:
    dyn_a, static = eqx.partition(mod_a, eqx.is_array)
    dyn_b, _ = eqx.partition(mod_b, eqx.is_array)
    dyn = jax.tree_map(
        lambda a, b: jnp.where(jnp.expand_dims(flag, tuple(range(1, a.ndim))), a, b),
        dyn_a,
        dyn_b,
    )
    return eqx.combine(dyn, static)
