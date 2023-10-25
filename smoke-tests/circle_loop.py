"""Example of using circle foraging environment"""


import enum
from typing import Any, Optional, Tuple

import numpy as np
import typer
from numpy.random import PCG64
from tqdm import tqdm

from emevo import _test_utils as test_utils
from emevo import make


class Rendering(str, enum.Enum):
    PYGAME = "pygame"
    MODERNGL = "moderngl"


class FoodNum(str, enum.Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    LOGISTIC = "logistic"


def main(
    steps: int = 100,
    render: Optional[Rendering] = None,
    food_initial_force: Tuple[float, float] = (0.0, 0.0),
    seed: int = 1,
    n_agents: int = 10,
    n_foods: int = 10,
    n_foods_later: int = 10,
    debug: bool = False,
    forward_sensor: bool = False,
    use_test_env: bool = False,
    obstacles: bool = False,
    angle: bool = False,
    env_shape: str = "square",
    food_loc_fn: str = "gaussian",
    food_num: FoodNum = FoodNum.CONSTANT,
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    if forward_sensor:
        env_kwargs: dict[str, Any] = {
            "sensor_range": (-30, 30),
            "sensor_length": 100,
            "foodloc_interval": 20,
        }
    else:
        env_kwargs = {"foodloc_interval": 20}

    if obstacles:
        env_kwargs["obstacles"] = [(100, 50, 100, 200)]

    if angle:
        env_kwargs["max_abs_angle"] = np.pi / 40

    env_kwargs["damping"] = 0.8

    if use_test_env:
        env = test_utils.predefined_env(**env_kwargs, seed=seed)
    else:
        env_kwargs["food_num_fn"] = food_num.value
        env_kwargs["food_loc_fn"] = food_loc_fn
        env = make(
            "CircleForaging-v0",
            env_shape=env_shape,
            food_initial_force=food_initial_force,
            seed=seed,
            n_initial_bodies=n_agents,
            **env_kwargs,
        )
    bodies = env.bodies()
    gen = np.random.Generator(PCG64(seed=seed))

    if render is not None:
        visualizer = env.visualizer(mode=render.value)
    else:
        visualizer = None

    change_foods = food_num is FoodNum.CONSTANT and n_foods_later != n_foods

    for i in tqdm(range(steps)):
        actions = {body: body.act_space.sample(gen) for body in bodies}
        # Samples for adding constant force for debugging
        # actions = {body: np.array([0.0, -1.0]) for body in bodies}
        _ = env.step(actions)  # type: ignore
        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()

        if change_foods and steps // 2 <= i:
            env.set_food_num_fn(("constant", n_foods_later))  # type: ignore


if __name__ == "__main__":
    typer.run(main)
