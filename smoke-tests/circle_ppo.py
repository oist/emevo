"""Example of using circle foraging environment"""

import datetime

import jax
import numpy as np
import optax
import qeuinox as eqx
import typer
from tqdm import tqdm

from emevo import Env, make
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.rl.ppo import NormalPPONet, Rollout, make_inormal


@eqx.filter_jit
def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Rollout, State, jax.Array]:
    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, jax.Array], Rollout]:
        state_t, obs_t = carried
        obs_t = obs_t.as_array()
        net_out = jax.vmap(network)(obs_t)
        actions = net_out.policy().sample(seed=key)
        state_t1, timestep = jax.vmap(env.step)(state_t, actions)
        rollout = Rollout(
            observations=obs_t,
            actions=actions,
            rewards=timestep.reward,
            terminations=1.0 - timestep.discount,
            values=net_out.value,
            means=net_out.mean,
            logstds=net_out.logstd,
        )
        return (state_t1, timestep.observation), rollout

    (state, obs), rollout = jax.lax.scan(
        step_rollout,
        (state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    next_value = jax.vmap(network.value)(obs.as_array())
    return state, rollout, obs, next_value


def run_training(
    key: jax.Array,
    n_agents: int,
    env: Env,
    adam: optax.GradientTransformation,
    n_total_steps: int,
    n_rollout_steps: int,
) -> NormalPPONet:
    assert n_agents == 1
    key, net_key, reset_key = jax.random.split(key, 3)
    pponet = jax.vmap(NormalPPONet)(jax.random.split(net_key, n_agents))
    adam_init, adam_update = adam
    opt_state = adam_init(eqx.filter(pponet, eqx.is_array))
    env_state, timestep = env.reset(reset_key)
    obs = timestep.observation

    n_loop = n_total_steps // n_rollout_steps
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
    env_shape: str = "circle",
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
    network = run_training(
        jax.random.PRNGKey(seed),
        n_agents,
        env,
        optax.adam(adam_lr, eps=adam_eps),
    )


if __name__ == "__main__":
    typer.run(main)
