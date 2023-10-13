from typing import Any

import chex
import jax
import jax.numpy as jnp

from emevo.environments.phyjax2d import ShapeDict, StateDict
from emevo.environments.phyjax2d_utils import circle_overwrap
from emevo.environments.utils.food_repr import ReprLocFn
from emevo.environments.utils.locating import Coordinate, InitLocFn


def _fail(*args, **kargs) -> jax.Array:
    return jnp.array([jnp.inf, jnp.inf])


def place_food(
    n_trial: int,
    food_radius: float,
    coordinate: Coordinate,
    reprloc_fn: ReprLocFn,
    reprloc_state: Any,
    key: chex.PRNGKey,
    shaped: ShapeDict,
    stated: StateDict,
) -> None:
    keys = jax.random.split(key, n_trial)
    loc_fn = jax.vmap(reprloc_fn, in_axes=(0, None), out_axes=(0, None))
    locations, state = loc_fn(keys, reprloc_state)
    radius = jnp.ones(n_trial) * food_radius
    ok = jnp.logical_and(
        jax.vmap(coordinate.contains_circle)(locations, radius),
        circle_overwrap(shaped, stated, locations, radius),
    )
    return jax.lax.cond(
        ok.any(),
        _fail,
    )
