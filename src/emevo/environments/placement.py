"""Place agent and food"""

import chex
import jax
import jax.numpy as jnp

from emevo.environments.phyjax2d import ShapeDict, StateDict
from emevo.environments.phyjax2d_utils import circle_overwrap
from emevo.environments.utils.food_repr import ReprLocFn, ReprLocState
from emevo.environments.utils.locating import Coordinate, InitLocFn

_vmap_co = jax.vmap(circle_overwrap, in_axes=(None, None, 0, 0))


def _place_common(
    coordinate: Coordinate,
    shaped: ShapeDict,
    stated: StateDict,
    locations: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    ok = jnp.logical_and(
        jax.vmap(coordinate.contains_circle)(locations, radius),
        jnp.logical_not(_vmap_co(shaped, stated, locations, radius)),
    )
    (ok_idx,) = jnp.nonzero(ok, size=1, fill_value=-1)
    ok_idx = ok_idx[0]
    return jax.lax.cond(
        ok_idx < 0,
        lambda: jnp.ones(2) * jnp.inf,
        lambda: locations[ok_idx],
    )


def place_food(
    n_trial: int,
    food_radius: float,
    coordinate: Coordinate,
    reprloc_fn: ReprLocFn,
    reprloc_state: ReprLocState,
    key: chex.PRNGKey,
    shaped: ShapeDict,
    stated: StateDict,
) -> jax.Array:
    """Returns `[inf, inf]` if it fails"""
    keys = jax.random.split(key, n_trial)
    loc_fn = jax.vmap(reprloc_fn, in_axes=(0, None))
    locations = loc_fn(keys, reprloc_state)
    return _place_common(
        coordinate,
        shaped,
        stated,
        locations,
        jnp.ones(n_trial) * food_radius,
    )


def place_agent(
    n_trial: int,
    agent_radius: float,
    coordinate: Coordinate,
    initloc_fn: InitLocFn,
    key: chex.PRNGKey,
    shaped: ShapeDict,
    stated: StateDict,
) -> jax.Array:
    """Returns `[inf, inf]` if it fails"""
    keys = jax.random.split(key, n_trial)
    locations = jax.vmap(initloc_fn)(keys)
    return _place_common(
        coordinate,
        shaped,
        stated,
        locations,
        jnp.ones(n_trial) * agent_radius,
    )
