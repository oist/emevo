"""Example of sexual reproduction in circle foraging environment"""
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
from emevo import make, utils


@dataclasses.dataclass
class SimpleContext:
    generation: int
    location: Vec2d


class HazardFn(str, enum.Enum):
    CONST = "const"
    GOMPERTZ = "gompertz"


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"


def birth_fn(
    status_a: bd.statuses.Status,
    status_b: bd.statuses.Status,
) -> float:
    avg_energy = (status_a.energy + status_b.energy) / 2.0
    return 1 / (1.0 + np.exp(-avg_energy))


def main(
    steps: int = 100,
    render: Rendering | None = None,
    food_initial_force: tuple[float, float] = (0.0, 0.0),
    agent_radius: float = 12.0,
    seed: int = 1,
    newborn_kind: str = "oviparous",
    hazard: HazardFn = HazardFn.CONST,
    debug: bool = False,
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    avg_lifetime = steps // 2

    if hazard == HazardFn.CONST:
        hazard_fn = bd.death.Deterministic(-10.0, avg_lifetime)
    elif hazard == HazardFn.GOMPERTZ:
        hazard_fn = bd.death.Gompertz(alpha=4e-5 / np.exp(1e-5 * avg_lifetime))
    else:
        raise ValueError(f"Invalid hazard {hazard}")
    birth_fn = bd.birth.Logistic(
        scale=0.1,
        alpha=0.1,
        beta_age=10.0 / avg_lifetime,
        age_delay=avg_lifetime / 4,
        energy_delay=0.0,
    )

    if newborn_kind == "oviparous":

        def produce_oviparous(_sa, _sb, encount: Encount) -> bd.Oviparous:
            loc = (encount.a.location() + encount.b.location()) * 0.5
            return bd.Oviparous(
                context=SimpleContext(encount.a.generation + 1, loc),
                time_to_birth=5,
            )

        manager = bd.SexualReprManager(
            initial_status_fn=partial(bd.statuses.Status, age=1, energy=0.0),
            hazard_fn=hazard_fn,
            birth_fn=birth_fn.sexual,
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
            initial_status_fn=partial(bd.statuses.Status, age=1, energy=0.0),
            hazard_fn=hazard_fn,
            birth_fn=birth_fn.sexual,
            produce_fn=produce_viviparous,
        )
    else:

        raise ValueError(f"Unknown newborn kind {newborn_kind}")

    env = make(
        "Forgaging-v0",
        food_initial_force=food_initial_force,
        agent_radius=agent_radius,
    )
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
            action_cost = np.linalg.norm(actions[body]) * 0.01
            observation = env.observe(body)
            energy_delta = observation.n_collided_foods - action_cost
            manager.update_status(body, energy_delta=energy_delta)
        _ = manager.reproduce(encounts)
        deads, newborns = manager.step()

        for dead in deads:
            print(f"{dead.body} is dead with {dead.status}")
            env.dead(dead.body)

        for context in map(operator.attrgetter("context"), newborns):
            loc = utils.sample_location(
                gen,
                context.location,
                radius_max=agent_radius * 3,
                radius_min=agent_radius * 1.5,
            )
            body = env.born(Vec2d(*loc), context.generation + 1)
            if body is not None:
                print(f"{body} was born")
                manager.register(body)
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