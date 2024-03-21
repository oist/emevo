""" Implementation of registry and built-in emevo environments.
"""

from emevo.environments.cf_with_smell import CircleForagingWithSmell
from emevo.environments.circle_foraging import CircleForaging
from emevo.environments.registry import register

register(
    "CircleForaging-v0",
    CircleForaging,
    "Phyjax2d circle foraging environment",
)

register(
    "CircleForaging-v1",
    CircleForagingWithSmell,
    "Phyjax2d circle foraging environment",
)
