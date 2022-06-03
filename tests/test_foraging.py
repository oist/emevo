import dataclasses
from typing import List

import numpy as np
import pytest
from loguru import logger
from numpy.testing import assert_almost_equal

from emevo import _test_utils as utils
from emevo.environments.pymunk_envs import Foraging

logger.enable("emevo")


@pytest.fixture
def env() -> Foraging:
    return utils.predefined_env()


@dataclasses.dataclass
class DebugLogHandler:
    logs: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        logger.add(self, format="{message}", level="DEBUG")

    def __call__(self, message: str) -> None:
        self.logs.append(message)

    def __contains__(self, query: str) -> bool:
        for log in self.logs:
            if query in log:
                return True
        return False

    def once(self, query: str) -> bool:
        n_occurance = 0
        for log in self.logs:
            if query in log:
                n_occurance += 1
        return n_occurance == 1


def test_eating(env: Foraging) -> None:
    """
    Confirm that eating detection (collision of an agent to a food) correctly works.
    Initially, food (F) and agent (A) are placed like
    A  F

    A  A
    , and we add force only to the lower right agent.
    """
    handler = DebugLogHandler()
    bodies = env.bodies()
    body = next(filter(lambda body: body.info().position.x > 100.0, bodies))
    already_ate = False

    while True:
        actions = {body: np.array([0.0, 1.0])}
        _ = env.step(actions)
        observation = env.observe(body)

        if already_ate:
            if "created" in handler:
                break
        else:
            if "eaten" in handler:
                assert_almost_equal(observation.collision[1], 1.0)
                already_ate = True
            else:
                assert observation.collision[1] == 0.0

    assert handler.once("eaten")
    assert handler.once("created")


def test_encounts(env: Foraging) -> None:
    """
    Confirm that encount detection (collision between agents) correctly works.
    Again, food (F) and agent (A) are placed like
    A  F

    A  A
    , and we add forces to two agents on the left, so that they collide.
    """
    body_higher, body_lower = None, None
    for body in env.bodies():
        pos = body.info().position
        if pos.x < 100 and 100 < pos.y:
            body_higher = body
        elif pos.x < 100 and pos.y < 100:
            body_lower = body

    assert body_higher is not None and body_lower is not None
    actions = {body_higher: np.array([0.0, -1.0]), body_lower: np.array([0.0, 1.0])}

    for i in range(100):
        encounts = env.step(actions)
        obs_high = env.observe(body_higher)
        obs_low = env.observe(body_lower)
        if len(encounts) > 0:
            assert_almost_equal(obs_low.collision[0], 1.0)
            assert_almost_equal(obs_high.collision[0], 1.0)
            assert len(encounts) == 1
            a, b = encounts[0]
            assert (body_higher is a and body_lower is b) or (
                body_lower is a and body_higher is b
            )
            break
        else:
            assert_almost_equal(obs_low.collision[0], 0.0)
            assert_almost_equal(obs_high.collision[0], 0.0)
