from typing import NamedTuple

import jax
import jax.numpy as jnp


class DiagonalNormal(NamedTuple):
    """Normal distribution with independent components.

    This is a conceptual clone of Distrax's ``Independent(LogStddevNormal)``
    specialized to the small API surface needed by Emevo.
    """

    loc: jax.Array
    log_scale: jax.Array

    def sample(self, *, seed: jax.Array) -> jax.Array:
        """Draw one sample with the same shape as the distribution parameters."""
        noise = jax.random.normal(seed, self.loc.shape, dtype=self.loc.dtype)
        return self.loc + jnp.exp(self.log_scale) * noise

    def log_prob(self, value: jax.Array) -> jax.Array:
        """Return the joint log probability over the final array dimension."""
        normalized = (value - self.loc) * jnp.exp(-self.log_scale)
        elementwise = -0.5 * (
            jnp.square(normalized) + 2.0 * self.log_scale + jnp.log(2.0 * jnp.pi)
        )
        return jnp.sum(elementwise, axis=-1)

    def entropy(self) -> jax.Array:
        """Return the joint entropy over the final array dimension."""
        elementwise = self.log_scale + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi))
        return jnp.sum(elementwise, axis=-1)
