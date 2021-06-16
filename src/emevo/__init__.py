"""
EmEvo is a simulation platform for embodied evolution of robots.
This package contains API definitions and some environment implementations.
"""
from emevo.admin import Admin, AsexualReprFn, IsDeadFn, SexualReprFn
from emevo.body import Body, Profile
from emevo.environment import Encount, Environment, make, register


__version__ = "0.1.0"
