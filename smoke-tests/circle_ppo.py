"""Example of using circle foraging environment"""

import datetime
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from tqdm import tqdm

from emevo import Env, Visualizer, make
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


def visualize(key: chex.PRNGKey, env: Env, network: NormalPPONet, n_steps: int) -> None:
    keys = jax.random.split(key, n_steps + 1)
    state, ts = env.reset(keys[0])
    obs = ts.obs
    visualizer = env.visualizer(state, figsize=(640.0, 640.0))

    @eqx.filter_jit
    def step(key: chex.PRNGKey, state: State, obs: Obs) -> tuple[State, Obs]:
        net_out = vmap_apply(network, obs.as_array())
        actions = net_out.policy().sample(seed=key)
        next_state, timestep = env.step(state, actions)
        return next_state, timestep.obs

    for key in keys[1:]:
        state, obs = step(key, state, obs)
        visualizer.render(state)
        visualizer.show()


def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Rollout, Obs, jax.Array]:
    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, Obs], Rollout]:
        state_t, obs_t = carried
        obs_t_array = obs_t.as_array()
        net_out = vmap_apply(network, obs_t_array)
        actions = net_out.policy().sample(seed=key)
        state_t1, timestep = env.step(state_t, actions)
        rewards = obs_t.collision[:, 1].astype(jnp.float32)
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


@eqx.filter_jit
def training_step(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    prng_key: jax.Array,
    n_rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    adam_update: optax.TransformUpdateFn,
    opt_state: optax.OptState,
    minibatch_size: int,
    n_optim_epochs: int,
) -> tuple[State, Obs, jax.Array, optax.OptState, NormalPPONet]:
    keys = jax.random.split(prng_key, N_MAX_AGENTS + 1)
    env_state, rollout, obs, next_value = exec_rollout(
        state,
        initial_obs,
        env,
        network,
        keys[0],
        n_rollout_steps,
    )
    batch = jax.jit(vmap_batch)(rollout, next_value, gamma, gae_lambda)
    opt_state, pponet = vmap_update(
        batch,
        network,
        adam_update,
        opt_state,
        keys[1:],
        minibatch_size,
        n_optim_epochs,
        0.2,
        0.01,
    )
    return env_state, obs, rollout.rewards, opt_state, pponet


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
    rewards = jnp.zeros(N_MAX_AGENTS)
    keys = jax.random.split(key, n_loop)
    for key in keys:
        env_state, obs, rewards_i, opt_state, pponet = training_step(
            env_state,
            obs,
            env,
            pponet,
            key,
            n_rollout_steps,
            gamma,
            gae_lambda,
            adam_update,
            opt_state,
            minibatch_size,
            n_optim_epochs,
        )
        ri = jnp.sum(rewards_i, axis=0)
        rewards = rewards + ri
        print(f"Rewards: {[x.item() for x in ri[: n_agents]]}")
    print(f"Sum of rewards {[x.item() for x in rewards[: n_agents]]}")
    return pponet


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
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
    minibatch_size: int = 128,
    n_rollout_steps: int = 512,
    n_total_steps: int = 512 * 100,
    food_loc_fn: str = "gaussian",
    env_shape: str = "square",
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
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(seed))
    network = run_training(
        train_key,
        n_agents,
        env,
        optax.adam(adam_lr, eps=adam_eps),
        gamma,
        gae_lambda,
        n_optim_epochs,
        minibatch_size,
        n_rollout_steps,
        n_total_steps,
    )
    if render:
        visualize(eval_key, env, network, 1000)
    eqx.tree_serialise_leaves("trained.eqx", network)


@app.command()
def vis(
    modelpath: Path = Path("trained.eqx"),
    n_steps: int = 1000,
    seed: int = 1,
    n_agents: int = 2,
    n_foods: int = 10,
    food_loc_fn: str = "gaussian",
    env_shape: str = "square",
    obstacles: str = "none",
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
    obs_space = env.obs_space.flatten()
    input_size = np.prod(obs_space.shape)
    act_size = np.prod(env.act_space.shape)
    net_key, eval_key = jax.random.split(jax.random.PRNGKey(seed))
    pponet = vmap_net(
        input_size,
        64,
        act_size,
        jax.random.split(net_key, N_MAX_AGENTS),
    )
    pponet = eqx.tree_deserialise_leaves(modelpath, pponet)
    visualize(eval_key, env, pponet, n_steps)


if __name__ == "__main__":
    app()
