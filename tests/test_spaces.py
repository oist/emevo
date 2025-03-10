from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal

from emevo.spaces import BoxSpace, DiscreteSpace, NamedTupleSpace


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def test_box(key: chex.PRNGKey) -> None:
    key1, key2 = jax.random.split(key)
    N = 5
    box_01 = BoxSpace(low=jnp.zeros(N), high=jnp.ones(N))
    sampled = box_01.sample(key1)
    assert 0 <= jnp.min(sampled) and jnp.max(sampled) <= 1.0
    assert box_01.is_bounded()
    unclipped = jnp.array([-1, 0, 0.5, 1, 2])
    clipped = box_01.clip(unclipped)
    assert_array_almost_equal(clipped, jnp.array([0, 0, 0.5, 1, 1]))
    assert box_01.contains(jnp.ones(N, dtype=jnp.float32) * 0.5)
    assert not box_01.contains(jnp.ones(N, dtype=jnp.float32) * 1.2)

    box_0_inf = BoxSpace(low=jnp.zeros(N), high=jnp.ones(N) * jnp.inf)
    sampled = box_0_inf.sample(key2)
    assert 0 <= jnp.min(sampled)
    assert not box_0_inf.is_bounded()
    clipped = box_0_inf.clip(unclipped)
    assert_array_almost_equal(clipped, jnp.array([0, 0, 0.5, 1, 2]))


def test_discrete(key: chex.PRNGKey) -> None:
    disc = DiscreteSpace(10)
    assert disc.contains(8)
    assert not disc.contains(-1)
    assert not disc.contains(10)
    sampled = disc.sample(key)
    assert 0 <= sampled and sampled < 10


def test_namedtuple(key: chex.PRNGKey) -> None:
    space1 = BoxSpace(low=jnp.zeros(10), high=jnp.ones(10))
    space2 = BoxSpace(low=jnp.ones(3) * 0.5, high=jnp.ones(3))

    class Observation(NamedTuple):
        sensor: jax.Array
        speed: jax.Array

    nt = NamedTupleSpace(Observation, sensor=space1, speed=space2)
    sampled: Observation = nt.sample(key)
    assert 0 <= jnp.min(sampled.sensor) and jnp.max(sampled.sensor) <= 1.0
    assert 0.5 <= jnp.min(sampled.speed) and jnp.max(sampled.speed) < 1.0

    assert nt.contains(
        (jnp.ones(10, dtype=jnp.float32) * 0.5, jnp.ones(3, dtype=jnp.float32) * 0.8),
    )
    assert not nt.contains(
        (jnp.ones(10, dtype=jnp.float32) * 0.5, jnp.ones(3, dtype=jnp.float32) * 1.2),
    )
    flattened = nt.flatten()
    assert flattened.shape == (13,)

    class ExtendedObs(NamedTuple):
        sensor: jax.Array
        speed: jax.Array
        angle: jax.Array

    space3 = BoxSpace(low=jnp.ones(1) * -3.14, high=jnp.ones(1) * 3.14)
    nt_extended = nt.extend(ExtendedObs, angle=space3)

    assert nt_extended.contains(
        (
            jnp.ones(10, dtype=jnp.float32) * 0.5,
            jnp.ones(3, dtype=jnp.float32) * 0.8,
            jnp.ones(1, dtype=jnp.float32) * 1.0,
        ),
    )
    assert not nt_extended.contains(
        (
            jnp.ones(10, dtype=jnp.float32) * 0.5,
            jnp.ones(3, dtype=jnp.float32) * 0.8,
            jnp.ones(1, dtype=jnp.float32) * 3.15,
        ),
    )

    assert nt_extended.flatten().shape == (14,)
