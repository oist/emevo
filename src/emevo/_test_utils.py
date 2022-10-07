"""
Common utilities used in smoke tests and unit tests.
Not expected to use externally.
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from emevo.environments import CircleForaging
from emevo.environments.utils.food_repr import ReprLoc, ReprNum
from emevo.environments.utils.locating import InitLoc


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
