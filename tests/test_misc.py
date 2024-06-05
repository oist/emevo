import chex
import jax
import jax.numpy as jnp

from emevo.environments.circle_foraging import _set_b2a
from emevo.environments.registry import _levenshtein_distance


def test_levenhtein() -> None:
    a = "ready to go"
    b = "readily to do"
    assert _levenshtein_distance(a, b) == 3


def test_set_b2a() -> None:
    key_a, key_b = jax.random.split(jax.random.PRNGKey(43), 2)
    a = jax.random.uniform(key_a, shape=(13, 2))
    b = jax.random.uniform(key_b, shape=(7, 2))
    flag_a = jnp.array(
        [
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
        ]
    )
    flag_b = jnp.array(
        [
            False,
            True,
            False,
            True,
            False,
            True,
            True,
        ]
    )
    c1 = a.at[flag_a].set(b[flag_b])
    c2 = _set_b2a(a, flag_a, b, flag_b)
    c3 = jax.jit(_set_b2a)(a, flag_a, b, flag_b)
    chex.assert_trees_all_close(c1, c2, c3)
