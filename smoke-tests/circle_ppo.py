"""Example of using circle foraging environment"""

import datetime

import jax
import numpy as np
import typer
from tqdm import tqdm

from emevo import make, env:
from emevo.rl.ppo import NormalPPONet


def run_training(key: jax.Array, n_agents: int, env: Env) -> NormalPPONet:
    key, net_key, reset_key = jax.random.split(key, 3)
    pponet = jax.vmap(NormalPPONet)(jax.random.split(net_key, n_agents))
    adam_init, adam_update = optax.adam(adam_lr, eps=adam_eps)
    opt_state = adam_init(eqx.filter(pponet, eqx.is_array))
    env_state, timestep = env.reset()
    obs = timestep.observation

    n_loop = n_total_steps // (n_agents * n_rollout_steps)
    return_reporting_interval = 1 if n_loop < 10 else n_loop // 10
    n_episodes, reward_sum = 0.0, 0.0
    for i in range(n_loop):
        key, rollout_key, update_key = jax.random.split(key, 3)
        env_state, rollout, obs, next_value = exec_rollout(
            env_state,
            obs,
            env,
            pponet,
            rollout_key,
            n_rollout_steps,
        )
        batch = make_batch(rollout, next_value, gamma, gae_lambda)
        opt_state, pponet = update_network(
            batch,
            pponet,
            adam_update,
            opt_state,
            update_key,
            minibatch_size,
            n_optim_epochs,
            ppo_clip_eps,
        )
        n_episodes += jnp.sum(rollout.terminations).item()
        reward_sum += jnp.sum(rollout.rewards).item()
        if i > 0 and (i % return_reporting_interval == 0):
            print(f"Mean episodic return: {reward_sum / n_episodes}")
            n_episodes = 0.0
            reward_sum = 0.0
    return pponet


def main(
    steps: int = 100,
    seed: int = 1,
    n_agents: int = 10,
    n_foods: int = 10,
    obstacles: str = "none",
    render: bool = False,
    replace: bool = False,
    adam_lr: float = 3e-4,
    adam_eps: float = 1e-7,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_optim_epochs: int = 10,
    minibatch_size: int = 1024,
    n_rollout_steps: int = 512,
    n_total_steps: int = 16 * 512 * 100,
    ppo_clip_eps: float = 0.2,
    food_loc_fn: str = "gaussian",
) -> None:
    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        n_max_agents=n_agents + 1,
        n_initial_agents=n_agents,
        food_num_fn=("constant", n_foods),
        food_loc_fn=food_loc_fn,
        foodloc_interval=20,
        obstacles=obstacles,
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
        state, _ = jit_step(state, jit_sample(key))
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
