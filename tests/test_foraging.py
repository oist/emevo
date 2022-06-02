import dataclasses
import enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
from loguru import logger

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
        def debug_only(record: Dict[str, Any]) -> None:
            return record["level"].name == "DEBUG"

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
    Confirm that eating detection (i.e., collision of an agent to a food) correctly
    works.
    Initially, food (F) and agent (A) are placed like
    A  F

    A  A
    , and we add force only to the lower right agent.
    """
    handler = DebugLogHandler()
    bodies = env.bodies()
    body = next(filter(lambda body: body._body.position.x >= 100.0, bodies))
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
                assert observation.collision[1] == 1.0
                already_ate = True
            else:
                assert observation.collision[1] == 0.0

    assert handler.once("eaten")
    assert handler.once("created")


def test_encounts(env: Foraging) -> None:
    """ """
    pass
