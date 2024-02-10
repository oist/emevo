from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from emevo.environments.circle_foraging import CircleForaging


class CFSObs(NamedTuple):
    """Observation of an agent with smell."""

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
                self.collision,
                self.velocity,
                jnp.expand_dims(self.angle, axis=1),
                jnp.expand_dims(self.angular_velocity, axis=1),
                jnp.expand_dims(self.energy, axis=1),
            ),
            axis=1,
        )


class CircleForagingWithSmell(CircleForaging):
    pass
