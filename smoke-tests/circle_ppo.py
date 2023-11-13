"""Example of using circle foraging environment"""

import datetime

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from tqdm import tqdm

from emevo import Env, make
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.rl.ppo_normal import (
    NormalPPONet,
    Rollout,
    make_inormal,
    vmap_apply,
    vmap_batch,
    vmap_net,
    vmap_update,
    vmap_value,
)

N_MAX_AGENTS: int = 10


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
    ) -> tuple[tuple[State, Obs], Rollout]:
        state_t, obs_t = carried
        obs_t_array = obs_t.as_array()
        net_out = vmap_apply(network, obs_t_array)
        actions = net_out.policy().sample(seed=key)
        state_t1, timestep = env.step(state_t, actions)
        rewards = obs_t.collision[:, 1]
        rollout = Rollout(
            observations=obs_t_array,
            actions=actions,
            rewards=rewards,
            terminations=jnp.zeros_like(rewards),
            values=net_out.value,
            means=net_out.mean,
            logstds=net_out.logstd,
        )
        return (state_t1, timestep.obs), rollout

    (state, obs), rollout = jax.lax.scan(
        step_rollout,
        (state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    next_value = vmap_value(network, obs.as_array()).ravel()
    return state, rollout, obs, next_value


def run_training(
    key: jax.Array,
    n_agents: int,
    env: Env,
    adam: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    n_optim_epochs: int,
    minibatch_size: int,
    n_rollout_steps: int,
    n_total_steps: int,
    ppo_clip_eps: float,
) -> NormalPPONet:
    key, net_key, reset_key = jax.random.split(key, 3)
    obs_space = env.obs_space.flatten()
    input_size = np.prod(obs_space.shape)
    act_size = np.prod(env.act_space.shape)
    pponet = vmap_net(
        input_size,
        64,
        act_size,
        jax.random.split(net_key, N_MAX_AGENTS),
    )
    adam_init, adam_update = adam
    opt_state = jax.vmap(adam_init)(eqx.filter(pponet, eqx.is_array))
    env_state, timestep = env.reset(reset_key)
    obs = timestep.obs

    n_loop = n_total_steps // n_rollout_steps
    return_reporting_interval = 1 if n_loop < 10 else n_loop // 10
    rewards = jnp.zeros(N_MAX_AGENTS)
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
        batch = jax.jit(vmap_batch)(rollout, next_value, gamma, gae_lambda)
        opt_state, pponet = eqx.filter_jit(vmap_update)(
            batch,
            pponet,
            adam_update,
            opt_state,
            jax.random.split(update_key, N_MAX_AGENTS),
            minibatch_size,
            n_optim_epochs,
            ppo_clip_eps,
            0.01,
        )
        rewards += rollout.rewards
    print(f"Sum of rewards {rewards}")
    return pponet


def main(
    seed: int = 1,
    n_agents: int = 2,
    n_foods: int = 10,
    obstacles: str = "none",
    render: bool = False,
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
    assert n_agents < N_MAX_AGENTS
    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        n_max_agents=N_MAX_AGENTS,
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
        gamma,
        gae_lambda,
        n_optim_epochs,
        minibatch_size,
        n_rollout_steps,
        n_total_steps,
        ppo_clip_eps,
    )


if __name__ == "__main__":
    typer.run(main)
