from emevo.environments.pymunk_envs.foraging import Foraging
from emevo.environments.registry import register

register("Foraging-v0", Foraging, "Pymunk circle foraging environment")
