"""Example of asexual reproduction in circle foraging environment"""
from __future__ import annotations

import enum
import operator
from functools import partial
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from numpy.random import PCG64
from pymunk.vec2d import Vec2d

from emevo import birth_and_death as bd
from emevo import make, utils
from emevo import visualizer as evis


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"
    HEADLESS = "headless"


class HazardFn(str, enum.Enum):
    CONST = "const"
    GOMPERTZ = "gompertz"


def main(
    steps: int = 100,
    render: Rendering | None = None,
    food_initial_force: tuple[float, float] = (0.0, 0.0),
    agent_radius: float = 12.0,
    seed: int = 1,
    hazard: HazardFn = HazardFn.CONST,
    debug: bool = False,
    video: Path | None = None,
) -> None:
    if debug:
        logger.enable("emevo")

    avg_lifetime = steps // 2
    if hazard == HazardFn.CONST:
        hazard_fn = bd.death.Deterministic(-10.0, avg_lifetime)
    elif hazard == HazardFn.GOMPERTZ:
        hazard_fn = bd.death.Gompertz(alpha=4e-5 / np.exp(1e-5 * avg_lifetime))
    else:
        raise ValueError(f"Invalid hazard {hazard}")
    birth_fn = bd.birth.Logistic(
        scale=10.0 / avg_lifetime,
        alpha=0.1,
        beta_age=10.0 / avg_lifetime,
        age_delay=avg_lifetime / 4,
        energy_delay=0.0,
    )
    exp_n_children = bd.population.expected_n_children(
        birth=birth_fn,
        hazard=hazard_fn,
        energy=1.0,
    )
    logger.info(f"Expected num. of children: {exp_n_children}")

    manager = bd.AsexualReprManager(
        initial_status_fn=partial(bd.Status, age=1, energy=4.0),
        hazard_fn=hazard_fn,
        birth_fn=birth_fn.asexual,
        produce_fn=lambda _, body: bd.Oviparous(
            parent=body,
            time_to_birth=5,
        ),
    )

    env = make("CircleForaging-v0", food_initial_force=food_initial_force)
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
            action_cost = np.linalg.norm(actions[body]) * 0.01
            observation = env.observe(body)
            energy_delta = observation.n_collided_foods - action_cost
            manager.update_status(body, energy_delta=energy_delta)
        manager.reproduce(bodies)
        deads, newborns = manager.step()

        for dead in deads:
            logger.info(f"{dead.body} is dead with {dead.status}")
            env.dead(dead.body)

        for newborn in newborns:
            loc = utils.sample_location(
                gen,
                newborn.location(),
                radius_max=agent_radius * 3,
                radius_min=agent_radius * 1.5,
            )
            body = env.born(Vec2d(*loc), newborn.parent.generation + 1)
            if body is not None:
                logger.info(f"{body} was born")
                manager.register(body)

        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()

        if env.is_extinct():
            logger.info(f"Extinct after {i} steps")
            break


if __name__ == "__main__":
    typer.run(main)
