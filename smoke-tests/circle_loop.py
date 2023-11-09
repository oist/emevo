"""Example of using circle foraging environment"""

import datetime

import jax
import numpy as np
import typer
from tqdm import tqdm

from emevo import make


def main(
    steps: int = 100,
    seed: int = 1,
    n_agents: int = 10,
    n_foods: int = 10,
    obstacles: str = "none",
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
        obstacles=obstacles,
        **additional_kwargs,
    )
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, steps + 1)
    state = env.reset(keys[0])

    if render:
        visualizer = env.visualizer(state)
    else:
        visualizer = None

    print(f"Avg. μs for step: {np.mean(elapsed_list)}")


if __name__ == "__main__":
    typer.run(main)
