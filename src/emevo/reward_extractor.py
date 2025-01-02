import dataclasses

import jax
import jax.numpy as jnp

from emevo.spaces import BoxSpace


@dataclasses.dataclass
class ActFoodExtractor:
    act_space: BoxSpace
    act_coef: float
    _max_norm: jax.Array = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._max_norm = jnp.sqrt(jnp.sum(self.act_space.high**2, axis=-1))

    def normalize_action(self, action: jax.Array) -> jax.Array:
        scaled = self.act_space.sigmoid_scale(action)
        norm = jnp.sqrt(jnp.sum(scaled**2, axis=-1, keepdims=True))
        return norm / self._max_norm

    def extract(self, ate_food: jax.Array, action: jax.Array) -> jax.Array:
        act_input = self.act_coef * self.normalize_action(action)
        return jnp.concatenate((ate_food.astype(jnp.float32), act_input), axis=1)


@dataclasses.dataclass
class SensorActFoodExtractor:
    act_space: BoxSpace
    act_coef: float
    sensor_indices: tuple | slice
    _max_norm: jax.Array = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._max_norm = jnp.sqrt(jnp.sum(self.act_space.high**2, axis=-1))

    def normalize_action(self, action: jax.Array) -> jax.Array:
        scaled = self.act_space.sigmoid_scale(action)
        norm = jnp.sqrt(jnp.sum(scaled**2, axis=-1, keepdims=True))
        return norm / self._max_norm

    def extract(
        self,
        ate_food: jax.Array,
        action: jax.Array,
        sensor_obs: jax.Array,
    ) -> jax.Array:
        # E.g., sensor with predator: (N_agents, N_sensors, N_obj)
        used_sensor_obs = sensor_obs[:, :, self.sensor_indices]
        avg_sensor_obs = jnp.mean(used_sensor_obs, axis=1)  # (N_agents, N_obj)
        act_input = self.act_coef * self.normalize_action(action)
        return jnp.concatenate(
            (ate_food.astype(jnp.float32), act_input, avg_sensor_obs),
            axis=1,
        )
