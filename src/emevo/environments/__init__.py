""" Implementation of registry and built-in emevo environments.
"""

from emevo.environments.registry import description, make, register
from emevo.environments.circle_foraging import CircleForaging

register(
    "CircleForaging-v0",
    CircleForaging,
    "Phyjax2d circle foraging environment",
)
