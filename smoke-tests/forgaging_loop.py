"""Example of using foraging environment"""

import enum
from typing import Optional, Tuple

import numpy as np
import typer
from numpy.random import PCG64

from emevo import make


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    OPENGL = "opengl"


def main(
    n_steps: int = 100,
    rendering: Optional[Rendering] = None,
    food_initial_force: Optional[Tuple[float, float]] = None,
    seed: int = 1,
) -> None:
    print(food_initial_force)
    env = make("Forgaging-v0", food_initial_force=food_initial_force)
    bodies = env.bodies()
    gen = np.random.Generator(PCG64(seed=seed))

    if rendering == Rendering.PYGAME:
        visualizer = env.visualizer(mode="pygame")
    else:
        visualizer = None

    for _ in range(n_steps):
        actions = {}
        for body in bodies:
            actions[body] = body.act_space.sample(gen)
            print(body._body.velocity)

        _encounts = env.step(actions)
        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()


if __name__ == "__main__":
    typer.run(main)