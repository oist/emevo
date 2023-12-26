"""
EmEvo is a simulation platform for embodied evolution of robots.
This package contains API definitions and some environment implementations.
"""


from emevo.env import Env, Status, TimeStep, UniqueID
from emevo.environments.registry import make, register
from emevo.vec2d import Vec2d
from emevo.visualizer import Visualizer

__version__ = "0.1.0"
