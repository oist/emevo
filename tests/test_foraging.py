import numpy as np
import pytest

from emevo.environments.pymunk_envs import Foraging
from emevo.environments.utils.locating import InitLoc


@pytest.fixture
def env() -> Foraging:
    locations = [
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    ]

    return Foraging(body_loc_fn=InitLoc.PRE_DIFINED(locations))


def test_encounts() -> None:
    env = Foraging()
