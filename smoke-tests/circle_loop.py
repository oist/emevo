"""Example of using circle foraging environment"""

import datetime
import enum
from typing import Any, Optional, Tuple

import chex
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
    n_agents: int = 10,
    n_foods: int = 10,
    forward_sensor: bool = False,
    use_test_env: bool = False,
    obstacles: bool = False,
    render: bool = False,
    replace: bool = False,
    fixed_agent_loc: bool = False,
    env_shape: str = "square",
    food_loc_fn: str = "gaussian",
) -> None:
    if fixed_agent_loc:
        additional_kwargs = {
            "agent_loc_fn": (
                "periodic",
                [40.0, 60.0],
                [60.0, 90.0],
                [80.0, 60.0],
                [100.0, 90.0],
                [120.0, 60.0],
            ),
        }
        n_agents = min(n_agents, 5)
    else:
        additional_kwargs = {}

    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        n_max_agents=n_agents + 10,
        n_initial_agents=n_agents,
        food_num_fn=("constant", n_foods),
        food_loc_fn=food_loc_fn,
        foodloc_interval=20,
        **additional_kwargs,
    )
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, steps + 1)
    state = env.reset(keys[0])

    if render:
        visualizer = env.visualizer(state)
    else:
        visualizer = None

    activate_index = n_agents
    jit_step = jax.jit(env.step)
    jit_sample = jax.jit(env.act_space.sample)
    elapsed_list = []
    for i, key in tqdm(zip(range(steps), keys[1:])):
        before = datetime.datetime.now()
        state = jit_step(state, jit_sample(key))
        elapsed = datetime.datetime.now() - before
        if i == 0:
            print(f"Compile: {elapsed.total_seconds()}s")
        elif i > 10:
            elapsed_list.append(elapsed / datetime.timedelta(microseconds=1))
        if replace and i % 1000 == 0:
            if n_agents + 5 <= activate_index:
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

    print(f"Avg. μs for step: {np.mean(elapsed_list)}")


if __name__ == "__main__":
    typer.run(main)
