""" Implementation of registry and built-in emevo environments.
"""

from emevo.environments.registry import register

register(
    "CircleForaging-v0",
    "emevo.environments.circle_foraging.CircleForaging",
    "Phyjax2d circle foraging environment",
)

register(
    "CircleForaging-v1",
    "emevo.environments.circle_foraging_with.CircleForaging",
    "Phyjax2d circle foraging environment",
)
