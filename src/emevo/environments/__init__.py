""" Implementation of registry and built-in emevo environments.
"""


from emevo.environments.pymunk_envs import foraging
from emevo.environments.pymunk_envs.foraging import Foraging
from emevo.environments.registry import description, make, register

register("Forgaging-v0", foraging.Foraging, "Pymunk circle foraging environment")
