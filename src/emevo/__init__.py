"""
EmEvo is a simulation platform for embodied evolution of robots.
This package contains API definitions and some environment implementations.
"""


from emevo.env import Profile, Env
from emevo.environments import make, register
from emevo.status import Status
from emevo.vec2d import Vec2d


__version__ = "0.1.0"
