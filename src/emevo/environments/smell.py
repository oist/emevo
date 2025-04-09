"""Common in smell"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from phyjax2d import State


class CFObsWithSmell(NamedTuple):
    """Observation of an agent."""

    sensor: jax.Array
    collision: jax.Array
    velocity: jax.Array
    angle: jax.Array
    angular_velocity: jax.Array
    energy: jax.Array
    smell: jax.Array

    def as_array(self) -> jax.Array:
        return jnp.concatenate(
            (
                self.sensor.reshape(self.sensor.shape[0], -1),
                self.collision.reshape(self.collision.shape[0], -1).astype(jnp.float32),
                self.velocity,
                jnp.expand_dims(self.angle, axis=1),
                jnp.expand_dims(self.angular_velocity, axis=1),
                jnp.expand_dims(self.energy, axis=1),
                self.smell,  # Assume multiple smell sources
            ),
            axis=1,
        )


def _compute_smell(
    decay_factor: float,
    front: bool,
    state: State,
    nose: jax.Array,
    center: jax.Array,
) -> jax.Array:
    # Compute distance
    dist = jnp.linalg.norm(state.p.xy - nose.reshape(1, 2), axis=1)
    smell = jnp.exp(-decay_factor * dist)
    if front:
        c2n = nose - center
        c2n_normal = c2n / jnp.clip(jnp.linalg.norm(c2n), a_min=1e-6)
        c2p = state.p.xy - center.reshape(1, 2)
        sep = jax.vmap(jnp.dot, in_axes=(0, None))(c2p, c2n_normal)
        is_front_and_active = jnp.logical_and(
            sep > 0,
            state.is_active,
        )
        return jnp.where(is_front_and_active, smell, 0.0)
    else:
        return jnp.where(state.is_active, smell, 0.0)


def _compute_smell_to_food(
    n_food_sources: int,
    front: bool,
    decay_factor: float,
    sc_state: State,
    nose: jax.Array,
    center: jax.Array,
) -> jax.Array:
    # Compute distance
    smell_masked = _compute_smell(decay_factor, front, sc_state, nose, center)
    smell_per_source = jnp.zeros(n_food_sources).at[sc_state.label].add(smell_masked)
    return smell_per_source


_vmap_compute_smell = jax.vmap(_compute_smell, in_axes=(None, None, None, 0, 0))
_vmap_compute_smell2f = jax.vmap(
    _compute_smell_to_food,
    in_axes=(None, None, None, None, 0, 0),
)
