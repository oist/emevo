from typing import NamedTuple

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from emevo.spaces import BoxSpace, DiscreteSpace, NamedTupleSpace


@pytest.fixture
def gen() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64())


def test_box(gen: np.random.Generator) -> None:
    N = 5
    box_01 = BoxSpace(low=np.zeros(N), high=np.ones(N))
    sampled = box_01.sample(gen)
    assert 0 <= np.min(sampled) and np.max(sampled) <= 1.0
    assert box_01.is_bounded()
    unclipped = np.array([-1, 0, 0.5, 1, 2])
    clipped = box_01.clip(unclipped)
    assert_array_almost_equal(clipped, np.array([0, 0, 0.5, 1, 1]))
    assert box_01.contains(np.ones(N, dtype=np.float32) * 0.5)
    assert not box_01.contains(np.ones(N, dtype=np.float64) * 0.5)
    assert not box_01.contains(np.ones(N, dtype=np.float32) * 1.2)

    box_0_inf = BoxSpace(low=np.zeros(N), high=np.ones(N) * np.inf)
    sampled = box_0_inf.sample(gen)
    assert 0 <= np.min(sampled)
    assert not box_0_inf.is_bounded()
    clipped = box_0_inf.clip(unclipped)
    assert_array_almost_equal(clipped, np.array([0, 0, 0.5, 1, 2]))


def test_discrete(gen: np.random.Generator) -> None:
    disc = DiscreteSpace(10)
    assert disc.contains(8)
    assert not disc.contains(-1)
    assert not disc.contains(10)
    sampled = disc.sample(gen)
    assert 0 <= sampled and sampled < 10


def test_namedtuple(gen: np.random.Generator) -> None:
    space1 = BoxSpace(low=np.zeros(10), high=np.ones(10))
    space2 = BoxSpace(low=np.ones(3) * 0.5, high=np.ones(3))

    class Observation(NamedTuple):
        sensor: NDArray
        speed: NDArray

    nt = NamedTupleSpace(Observation, sensor=space1, speed=space2)
    sampled: Observation = nt.sample(gen)
    assert 0 <= np.min(sampled.sensor) and np.max(sampled.sensor) <= 1.0
    assert 0.5 <= np.min(sampled.speed) and np.max(sampled.speed) < 1.0

    assert nt.contains(
        (np.ones(10, dtype=np.float32) * 0.5, np.ones(3, dtype=np.float32) * 0.8),
    )
    assert not nt.contains(
        (np.ones(10, dtype=np.float32) * 0.5, np.ones(3, dtype=np.float32) * 1.2),
    )
    flattened = nt.flatten()
    assert flattened.shape == (13,)
