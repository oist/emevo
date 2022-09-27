"""Example of using foraging environment with AsexualReprManager """
from __future__ import annotations

import dataclasses
import enum
import operator
from functools import partial
from pathlib import Path

import numpy as np
import typer
from numpy.random import PCG64
from pymunk.vec2d import Vec2d

from emevo import birth_and_death as bd
from emevo import make, utils
from emevo import visualizer as evis


@dataclasses.dataclass
class SimpleContext:
    generation: int
    location: Vec2d


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"
    HEADLESS = "headless"


class HazardFn(str, enum.Enum):
    CONST = "const"
    GOMPERTZ = "gompertz"
    WEIBULL = "weibull"


def main(
    steps: int = 100,
    render: Rendering | None = None,
    food_initial_force: tuple[float, float] = (0.0, 0.0),
    agent_radius: float = 12.0,
    seed: int = 1,
    hazard: HazardFn = HazardFn.CONST,
    birth_rate: float = 0.001,
    debug: bool = False,
    video: Path | None = None,
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    avg_lifetime = steps // 2
    if hazard == HazardFn.CONST:
        hazard_fn = bd.death.Deterministic(-10.0, avg_lifetime)
    elif hazard == HazardFn.GOMPERTZ:
        hazard_fn = bd.death.Gompertz(beta=np.log(10) / avg_lifetime)
    elif hazard == HazardFn.WEIBULL:
        alpha = 0.5 * (1.0 / avg_lifetime)
        hazard_fn = bd.death.Weibull(alpha1=alpha, alpha2=alpha, beta=1.0)
    else:
        assert False

    manager = bd.AsexualReprManager(
        initial_status_fn=partial(bd.statuses.AgeAndEnergy, age=1, energy=0.0),
        hazard_fn=hazard_fn,
        birth_fn=lambda _: birth_rate,
        produce_fn=lambda _, body: bd.Oviparous(
            context=SimpleContext(body.generation + 1, body.location()),
            time_to_birth=5,
        ),
    )

    env = make("Forgaging-v0", food_initial_force=food_initial_force)
    manager.register(env.bodies())
    gen = np.random.Generator(PCG64(seed=seed))

    if render is not None:
        if render == Rendering.HEADLESS:
            visualizer = env.visualizer(mode="moderngl", mgl_backend="headless")
        else:
            visualizer = env.visualizer(mode=render.value)
        if video is not None:
            visualizer = evis.SaveVideoWrapper(visualizer, video, fps=60)
    else:
        visualizer = None

    for i in range(steps):
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

        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()

        if env.is_extinct():
            print(f"Extinct after {i} steps")
            break


if __name__ == "__main__":
    typer.run(main)
