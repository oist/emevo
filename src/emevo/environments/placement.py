"""Place agent and food"""

import chex
import jax
import jax.numpy as jnp

from emevo.environments.locating import Coordinate, LocatingFn, LocatingState
from emevo.environments.phyjax2d import ShapeDict, StateDict
from emevo.environments.phyjax2d_utils import circle_overwrap

_vmap_co = jax.vmap(circle_overwrap, in_axes=(None, None, 0, None))


def place(
    n_trial: int,
    radius: float,
    coordinate: Coordinate,
    loc_fn: LocatingFn,
    loc_state: LocatingState,
    key: chex.PRNGKey,
    shaped: ShapeDict,
    stated: StateDict,
) -> jax.Array:
    """Returns `[inf, inf]` if it fails"""
    keys = jax.random.split(key, n_trial)
    vmap_loc_fn = jax.vmap(loc_fn, in_axes=(0, None))
    locations = vmap_loc_fn(keys, loc_state)
    contains_fn = jax.vmap(coordinate.contains_circle, in_axes=(0, None))
    ok = jnp.logical_and(
        contains_fn(locations, radius),
        jnp.logical_not(_vmap_co(shaped, stated, locations, radius)),
    )
    (ok_idx,) = jnp.nonzero(ok, size=1, fill_value=-1)
    ok_idx = ok_idx[0]
    return jax.lax.cond(
        ok_idx < 0,
        lambda: jnp.ones(2) * jnp.inf,
        lambda: locations[ok_idx],
    )
