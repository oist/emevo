"""
EmEvo is a simulation platform for embodied evolution of robots.
This package contains API definitions and some environment implementations.
"""
# birth_and_death is an optional API, so is not imported by default
from emevo.body import Body, Encount, Profile  # noqa
from emevo.environment import Environment  # noqa
from emevo.environments import make, register  # noqa

__version__ = "0.1.0"
