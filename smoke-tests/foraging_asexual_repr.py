"""Example of using foraging environment"""
from __future__ import annotations

import dataclasses
import enum
import operator
from functools import partial

import numpy as np
import typer
from numpy.random import PCG64
from pymunk.vec2d import Vec2d

from emevo import birth_and_death as bd
from emevo import make


def sigmoid(x: float) -> float:
    return x / (1.0 + abs(x))


@dataclasses.dataclass
class SimpleContext:
    generation: int
    location: Vec2d


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"


def main(
    steps: int = 100,
    render: Rendering | None = None,
    food_initial_force: tuple[float, float] = (0.0, 0.0),
    seed: int = 1,
    debug: bool = False,
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    manager = bd.AsexualReprManager(
        initial_status_fn=partial(bd.statuses.AgeAndEnergy, age=1, energy=0.0),
        death_prob_fn=bd.death.hunger_or_infirmity(-10.0, 1000.0),
        success_prob=lambda status: sigmoid(status.energy),
        produce=lambda _, body: bd.Oviparous(
            context=SimpleContext(body.generation + 1, body.location()),
            time_to_birth=5,
        ),
    )

    env = make("Forgaging-v0", food_initial_force=food_initial_force)
    manager.register(env.bodies())
    gen = np.random.Generator(PCG64(seed=seed))

    if render is not None:
        visualizer = env.visualizer(mode=render.value)
    else:
        visualizer = None

    for _ in range(steps):
        bodies = env.bodies()
        actions = {body: body.act_space.sample(gen) for body in bodies}
        _ = env.step(actions)
        for body in bodies:
            manager.update_status(body, energy_update=gen.normal(loc=0.0, scale=0.1))
        _ = manager.reproduce(bodies)
        deads, newborns = manager.step()

        for dead in deads:
            print(f"{dead.body} is dead with {dead.status}")
            env.dead(dead.body)

        for context in map(operator.attrgetter("context"), newborns):
            body = env.born(context.location, context.generation + 1)
            if body is not None:
                manager.register(body)

        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()

        if env.is_extinct():
            print("Extinct")
            break


if __name__ == "__main__":
    typer.run(main)
