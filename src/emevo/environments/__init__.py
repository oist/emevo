""" Implementation of registry and built-in emevo environments.
"""

from emevo.environments.circle_foraging import CircleForaging
from emevo.environments.registry import description, make, register

register(
    "CircleForaging-v0",
    CircleForaging,
    "Phyjax2d circle foraging environment",
)
