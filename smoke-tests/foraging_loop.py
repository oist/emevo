"""Example of using foraging environment"""
from __future__ import annotations

import enum

import numpy as np
import typer
from numpy.random import PCG64

from emevo import _test_utils as test_utils
from emevo import make


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"


def main(
    steps: int = 100,
    render: Rendering | None = None,
    food_initial_force: tuple[float, float] = (0.0, 0.0),
    seed: int = 1,
    debug: bool = False,
    forward_sensor: bool = False,
    use_test_env: bool = False,
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    if forward_sensor:
        env_kwargs = {"sensor_range": (-60, 60), "sensor_length": 16}
    else:
        env_kwargs = {}

    if use_test_env:
        env = test_utils.predefined_env(**env_kwargs)
    else:
        env = make("Forgaging-v0", food_initial_force=food_initial_force, **env_kwargs)
    bodies = env.bodies()
    gen = np.random.Generator(PCG64(seed=seed))

    if render is not None:
        visualizer = env.visualizer(mode=render.value)
    else:
        visualizer = None

    for _ in range(steps):
        actions = {body: body.act_space.sample(gen) for body in bodies}
        # Samples for adding constant force for debugging
        # actions = {body: np.array([0.0, -1.0]) for body in bodies}
        _ = env.step(actions)
        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()


if __name__ == "__main__":
    typer.run(main)