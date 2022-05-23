"""Example of using foraging environment"""

import enum

from typing import Optional

import numpy as np
import typer

from emevo import make


class Rendering(str, enum.Enum):
    MPL = "mpl"
    OPENGL = "opengl"


def main(
    n_steps: int = 100,
    rendering: Optional[Rendering] = None,
    seed: int = 1,
) -> None:
    env = make("Forgaging-v0")
    bodies = env.bodies()
    gen = np.random.Generator(seed=seed)

    if rendering == Rendering.MPL:
        visualizer = env.visualizer()
        visualizer.show()
    else:
        visualizer = None

    for _ in range(n_steps):
        actions = {}
        for body in bodies:
            actions[body] = body.act_space.sample(gen)

        _encounts = env.step(actions)
        if visualizer is not None:
            visualizer.render(env)


if __name__ == "__main__":
    typer.run(main)
