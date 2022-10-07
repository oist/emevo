""" Implementation of registry and built-in emevo environments.
"""


from emevo.environments.pymunk_envs import circle
from emevo.environments.pymunk_envs.circle import CircleForaging
from emevo.environments.registry import description, make, register

register("Forgaging-v0", circle.CircleForaging, "Pymunk circle foraging environment")
