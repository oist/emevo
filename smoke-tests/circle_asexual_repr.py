"""Example of asexual reproduction in circle foraging environment"""
from __future__ import annotations

import enum
import sys
from functools import partial
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from numpy.random import PCG64
from pymunk.vec2d import Vec2d

from emevo import Status
from emevo import birth_and_death as bd
from emevo import make
from emevo import visualizer as evis
from emevo._test_utils import sample_location


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
    n_agent_sensors: int = 8,
    sensor_length: float = 10.0,
    env_shape: str = "square",
    seed: int = 1,
    hazard: HazardFn = HazardFn.CONST,
    debug: bool = False,
    video: Path | None = None,
) -> None:
    if debug:
        logger.enable("emevo")
    logger.add(
        sys.stderr,
        filter="__main__",
        level="DEBUG" if debug else "INFO",
    )

    avg_lifetime = steps // 2
    if hazard == HazardFn.CONST:
        hazard_fn = bd.death.Constant(
            alpha_const=1.0 / avg_lifetime,
            alpha_energy=1.0 / avg_lifetime,
            gamma=0.1,
        )
    elif hazard == HazardFn.GOMPERTZ:
        hazard_fn = bd.death.Gompertz(
            alpha_const=1.0 / avg_lifetime,
            alpha_energy=1.0 / avg_lifetime,
            gamma=0.1,
            beta=1e-4,
        )
    else:
        raise ValueError(f"Invalid hazard {hazard}")
    birth_fn = bd.birth.Logistic(
        scale=10.0 / avg_lifetime,
        alpha=0.1,
        beta=10.0 / avg_lifetime,
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
        initial_status_fn=partial(Status, age=1, energy=4.0),
        hazard_fn=hazard_fn,
        birth_fn=birth_fn.asexual,
        produce_fn=lambda status, body: bd.Oviparous(
            parent=body,
            parental_status=status,
            time_to_birth=5,
        ),
    )

    env = make(
        "CircleForaging-v0",
        food_initial_force=food_initial_force,
        n_agent_sensors=n_agent_sensors,
        sensor_length=sensor_length,
        env_shape=env_shape,
    )
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
            env.remove_body(dead.body)

        for newborn in newborns:
            loc = sample_location(
                gen,
                newborn.location(),
                radius_max=agent_radius * 3,
                radius_min=agent_radius * 1.5,
            )
            body = env.locate_body(Vec2d(*loc), newborn.parent.generation + 1)
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
