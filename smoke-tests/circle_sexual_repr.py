"""Example of sexual reproduction in circle foraging environment"""
from __future__ import annotations

import dataclasses
import enum
import sys
from functools import partial

import numpy as np
import typer
from loguru import logger
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


def birth_fn(status_a: bd.Status, status_b: bd.Status) -> float:
    avg_energy = (status_a.energy + status_b.energy) / 2.0
    return 1 / (1.0 + np.exp(-avg_energy))


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
) -> None:
    logger.remove()
    if debug:
        logger.enable("emevo")
    logger.add(
        sys.stderr,
        filter="__main__",
        level="DEBUG" if debug else "INFO",
    )

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

    def produce(_sa, _sb, encount: Encount) -> bd.Oviparous:
        return bd.Oviparous(
            parent=encount.a,
            time_to_birth=5,
        )

    manager = bd.SexualReprManager(
        initial_status_fn=partial(bd.Status, age=1, energy=0.0),
        hazard_fn=hazard_fn,
        birth_fn=birth_fn.sexual,
        produce_fn=produce,
    )

    env = make(
        "CircleForaging-v0",
        food_initial_force=food_initial_force,
        agent_radius=agent_radius,
        n_agent_sensors=n_agent_sensors,
        sensor_length=sensor_length,
        env_shape=env_shape,
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
        logger.debug("Step start")
        encounts = env.step(actions)
        logger.debug("Step end")
        for body in bodies:
            action_cost = np.linalg.norm(actions[body]) * 0.01
            logger.debug("Observe start")
            observation = env.observe(body)
            logger.debug("Observe end")
            energy_delta = observation.n_collided_foods - action_cost
            manager.update_status(body, energy_delta=energy_delta)
        _ = manager.reproduce(encounts)
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
            if body is not None:
                manager.register(body)

        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()

        if env.is_extinct():
            logger.info(f"Extinct after {i} steps")
            break


if __name__ == "__main__":
    typer.run(main)
