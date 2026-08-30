import jax
import jax.numpy as jnp
import numpy as np

from emevo.rl.prob_dist import DiagonalNormal


def test_standard_normal_log_prob_and_entropy() -> None:
    distribution = DiagonalNormal(jnp.zeros((2, 3)), jnp.zeros((2, 3)))

    expected_log_prob = -1.5 * jnp.log(2.0 * jnp.pi)
    expected_entropy = 1.5 * (1.0 + jnp.log(2.0 * jnp.pi))

    np.testing.assert_allclose(
        distribution.log_prob(jnp.zeros((2, 3))),
        jnp.full((2,), expected_log_prob),
    )
    np.testing.assert_allclose(
        distribution.entropy(),
        jnp.full((2,), expected_entropy),
    )


def test_sample_is_jittable_and_has_parameter_shape() -> None:
    distribution = DiagonalNormal(jnp.ones((2, 3)), jnp.zeros((2, 3)))
    sample = jax.jit(lambda key: distribution.sample(seed=key))(jax.random.PRNGKey(0))

    assert sample.shape == (2, 3)
