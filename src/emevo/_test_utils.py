"""
Common utilities used in smoke tests and unit tests.
Not expected to use externally.
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from emevo import Body, spaces
from emevo.environments import CircleForaging
from emevo.environments.utils.food_repr import ReprLoc, ReprNum
from emevo.environments.utils.locating import InitLoc


class FakeBody(Body):
    def __init__(self, act_dim: int = 1, obs_dim: int = 1) -> None:
        act_space = spaces.BoxSpace(
            np.zeros(act_dim, dtype=np.float32),
            np.ones(act_dim, dtype=np.float32),
        )
        obs_space = spaces.BoxSpace(
            np.zeros(obs_dim, dtype=np.float32),
            np.ones(obs_dim, dtype=np.float32),
        )
        super().__init__(act_space, obs_space)

    def location(self) -> NDArray:
        return np.array(())


def predefined_env(
    *,
    agent_locations: list[NDArray] | None = None,
    food_locations: list[NDArray] | None = None,
    **kwargs,
) -> CircleForaging:
    if agent_locations is None:
        agent_locations = [
            np.array([50, 60]),
            np.array([50, 140]),
            np.array([150, 40]),
        ]
    if food_locations is None:
        food_locations = [np.array([150, 160])]

    return CircleForaging(
        n_initial_bodies=len(agent_locations),
        body_loc_fn=InitLoc.PRE_DIFINED(agent_locations),
        food_num_fn=ReprNum.CONSTANT(len(food_locations)),
        food_loc_fn=ReprLoc.PRE_DIFINED(itertools.cycle(food_locations)),
        **kwargs,
    )


def sample_location(
    gen: Generator,
    center: tuple[float, float],
    radius_max: float,
    radius_min: float = 0.0,
) -> tuple[float, float]:
    cx, cy = center
    theta = gen.random() * 2.0 * np.pi
    radius_range = radius_max - radius_min
    radius = np.sqrt(gen.random()) * radius_range + radius_min
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    return x, y
