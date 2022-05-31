import pytest

from emevo import _test_utils as utils
from emevo.environments.pymunk_envs import Foraging


@pytest.fixture
def env() -> Foraging:
    return utils.predefined_env()


def test_encounts(env: Foraging) -> None:
    pass


def test_eating(env: Foraging) -> None:
    pass
