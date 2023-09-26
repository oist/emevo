"""
EmEvo is a simulation platform for embodied evolution of robots.
This package contains API definitions and some environment implementations.
"""


from emevo.env import Profile, Env
from emevo.environments import make, register
from emevo.status import Status


def __disable_loguru() -> None:
    from loguru import logger

    logger.disable("emevo")


__disable_loguru()
__version__ = "0.1.0"
