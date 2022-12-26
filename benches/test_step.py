"""Example of using circle foraging environment"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.random import PCG64

from emevo import Env, make


def make_env(threaded: bool, population: int) -> Env:
    return make(
        "CircleForaging-v0",
        env_shape="square",
        n_initial_bodies=population,
        n_agent_sensors=24,
        sensor_length=40.0,
        xlim=(0.0, 360.0),
        ylim=(0.0, 360.0),
        threaded=threaded,
    )


@pytest.mark.parametrize(
    "threaded, population",
    [(False, 10), (True, 10), (False, 40), (True, 40)],
)
def test_step(benchmark, threaded: bool, population: int) -> None:
    env = make_env(threaded=threaded, population=population)
    bodies = env.bodies()
    gen = np.random.Generator(PCG64(seed=1))

    actions = {body: body.act_space.sample(gen) for body in bodies}
    _ = benchmark(env.step, actions)


@pytest.mark.parametrize(
    "threaded, population",
    [(False, 10), (True, 10), (False, 40), (True, 40)],
)
def test_observe(benchmark, threaded: bool, population: int) -> None:
    env = make_env(threaded=threaded, population=population)
    bodies = env.bodies()
    gen = np.random.Generator(PCG64(seed=1))

    actions = {body: body.act_space.sample(gen) for body in bodies}
    _ = env.step(actions)
    bodies_iter = itertools.cycle(bodies)

    def observe():
        return env.observe(next(bodies_iter))

    _ = benchmark(observe)
