"""Example of using circle foraging environment"""

import chex
import enum
from typing import Any, Optional, Tuple

import jax
import numpy as np
import typer
from numpy.random import PCG64
from tqdm import tqdm

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
        n_max_agents=20,
        n_initial_agents=6,
        **env_kwargs,
    )
    key = jax.random.PRNGKey(43)
    keys = jax.random.split(key, steps + 1)
    state = env.reset(keys[0])

    if render:
        visualizer = env.visualizer(state)
    else:
        visualizer = None

    activate_index = 5
    jit_step = jax.jit(env.step)
    jit_sample = jax.jit(env.act_space.sample)
    for i, key in tqdm(zip(range(steps), keys[1:])):
        # key, act_key = jax.random.split(state.key)
        # state = state.replace(key=key)
        act = jit_sample(key)
        state = jit_step(state, act)
        if i % 1000 == 0:
            if 10 <= activate_index:
                state, success = env.deactivate(state, activate_index)
                if not success:
                    print(f"Failed to deactivate agent! {activate_index}")
                else:
                    activate_index -= 1
            else:
                state, success = env.activate(state, 0)
                if not success:
                    print("Failed to activate agent!")
                else:
                    activate_index += 1

        if visualizer is not None:
            visualizer.render(state)
            visualizer.show()


if __name__ == "__main__":
    typer.run(main)
