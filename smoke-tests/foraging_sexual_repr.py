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

from emevo import Encount
from emevo import birth_and_death as bd
from emevo import make


@dataclasses.dataclass
class SimpleContext:
    generation: int
    location: Vec2d


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"


def success_prob_fn(
    status_a: bd.statuses.HasEnergy,
    status_b: bd.statuses.HasEnergy,
) -> float:
    avg_energy = (status_a.energy + status_b.energy) / 2.0
    return 1 / (1.0 + np.exp(-avg_energy))


def main(
    steps: int = 100,
    render: Rendering | None = None,
    food_initial_force: tuple[float, float] = (0.0, 0.0),
    seed: int = 1,
    newborn_kind: str = "oviparous",
    debug: bool = False,
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    if newborn_kind == "oviparous":

        def produce_oviparous(_sa, _sb, encount: Encount) -> bd.Oviparous:
            loc = encount.a.location()
            return bd.Oviparous(
                context=SimpleContext(encount.a.generation + 1, loc),
                time_to_birth=5,
            )

        manager = bd.SexualReprManager(
            initial_status_fn=partial(bd.statuses.AgeAndEnergy, age=1, energy=0.0),
            death_prob_fn=bd.death.hunger_or_infirmity(-10.0, 1000.0),
            success_prob_fn=success_prob_fn,
            produce_fn=produce_oviparous,
        )

    elif newborn_kind == "viviparous":

        def produce_viviparous(_sa, _sb, encount: Encount) -> bd.Viviparous:
            loc = encount.a.location()
            return bd.Viviparous(
                context=SimpleContext(encount.a.generation + 1, loc),
                parent=encount.a,
                time_to_birth=5,
            )

        manager = bd.SexualReprManager(
            initial_status_fn=partial(bd.statuses.AgeAndEnergy, age=1, energy=0.0),
            death_prob_fn=bd.death.hunger_or_infirmity(-10.0, 1000.0),
            success_prob_fn=success_prob_fn,
            produce_fn=produce_viviparous,
        )
    else:

        raise ValueError(f"Unknown newborn kind {newborn_kind}")

    env = make("Forgaging-v0", food_initial_force=food_initial_force)
    manager.register(env.bodies())
    gen = np.random.Generator(PCG64(seed=seed))

    if render is not None:
        visualizer = env.visualizer(mode=render.value)
    else:
        visualizer = None

    for i in range(steps):
        bodies = env.bodies()
        actions = {body: body.act_space.sample(gen) for body in bodies}
        encounts = env.step(actions)
        for body in bodies:
            manager.update_status(body, energy_update=gen.normal(loc=0.0, scale=0.1))
        _ = manager.reproduce(encounts)
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
            print(f"Extinct after {i} steps")
            break


if __name__ == "__main__":
    typer.run(main)
