"""Example of using circle foraging environment"""

import datetime

import jax
import jax.numpy as jnp
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
    food_num_fn: str = "constant",
    food_growth_rate: float = 0.1,
    xlim: float = 200.0,
    ylim: float = 200.0,
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

    n_max_agents = n_agents + 10
    if food_num_fn == "constant":
        fnf = "constant", n_foods
    elif food_num_fn == "logistic":
        fnf = "logistic", n_foods, food_growth_rate, n_foods * 2
    else:
        raise ValueError(f"Invalid food_num_fn: {food_num_fn}")
    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        n_max_agents=n_max_agents,
        n_initial_agents=n_agents,
        food_num_fn=fnf,
        food_loc_fn=food_loc_fn,
        foodloc_interval=20,
        obstacles=obstacles,
        **additional_kwargs,
        xlim=(0.0, 400.0),
        ylim=(0.0, 400.0),
    )
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, steps + 1)
    state, _ = env.reset(keys[0])

    if render:
        visualizer = env.visualizer(state)
    else:
        visualizer = None

    jit_step = jax.jit(env.step)
    jit_sample = jax.jit(
        lambda key: jax.vmap(env.act_space.sample)(jax.random.split(key, n_max_agents))
    )
    elapsed_list = []

    replace_interval = steps // 10
    deactivate_index = n_agents - 1
    activate_p = jnp.zeros(n_max_agents).at[jnp.arange(5)].set(0.5)
    deactivate = jax.jit(env.deactivate)
    activate = jax.jit(env.activate)
    for i in tqdm(range(steps)):
        before = datetime.datetime.now()
        act = jit_sample(keys[i + 1])
        state, _ = jit_step(state, act)
        elapsed = datetime.datetime.now() - before
        if i == 0:
            print(f"Compile: {elapsed.total_seconds()}s")
        elif i > 10:
            elapsed_list.append(elapsed / datetime.timedelta(microseconds=1))

        if replace and i % replace_interval == 0:
            if i < steps // 2:
                flag = (
                    jnp.zeros(n_max_agents, dtype=bool).at[deactivate_index].set(True)
                )
                state = deactivate(state, flag)
                deactivate_index -= 1
            else:
                flag = jax.random.bernoulli(keys[i + 1], p=activate_p)
                state, parents = activate(state, flag)
                print("Parents: ", parents)

        if visualizer is not None:
            visualizer.render(state.physics)
            visualizer.show()

    print(f"Avg. μs for step: {np.mean(elapsed_list)}")


if __name__ == "__main__":
    typer.run(main)
