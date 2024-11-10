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
    "emevo.environments.circle_foraging_with_neurotoxin.CircleForagingWithNeurotoxin",
    "Phyjax2d circle foraging environment with neuro toxin",
)
