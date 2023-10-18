"""Example of using circle foraging environment"""


import enum
from typing import Any, Optional, Tuple

import numpy as np
import typer
from numpy.random import PCG64
from tqdm import tqdm
import jax
from emevo import make




class FoodNum(str, enum.Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    LOGISTIC = "logistic"


def main(
    steps: int = 100,
    seed: int = 1,
    n_foods: int = 10,
    n_foods_later: int = 10,
    debug: bool = False,
    forward_sensor: bool = False,
    use_test_env: bool = False,
    obstacles: bool = False,
    angle: bool = False,
    render: bool = False,
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

    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        **env_kwargs,
    )
    state = env.reset(jax.random.PRNGKey(43))

    if render is not None:
        visualizer = env.visualizer()

    change_foods = food_num is FoodNum.CONSTANT and n_foods_later != n_foods

    for i in tqdm(range(steps)):
        # actions = {body: body.act_space.sample(gen) for body in bodies}
        # Samples for adding constant force for debugging
        # actions = {body: np.array([0.0, -1.0]) for body in bodies}
        # _ = env.step(actions)  # type: ignore
        if visualizer is not None:
            visualizer.render(env)
            visualizer.show()



if __name__ == "__main__":
    typer.run(main)
