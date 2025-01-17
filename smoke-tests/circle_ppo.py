"""Example of using circle foraging environment"""

import dataclasses
from pathlib import Path

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from numpy.typing import NDArray
from serde import toml

from emevo import Env, make
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.exp_utils import CfConfig
from emevo.rl.ppo_normal import (
    NormalPPONet,
    Rollout,
    vmap_apply,
    vmap_batch,
    vmap_net,
    vmap_update,
    vmap_value,
)
from emevo.visualizer import SaveVideoWrapper

PROJECT_ROOT = Path(__file__).parent.parent
N_MAX_AGENTS: int = 20


def weight_summary(network):
    params, _ = eqx.partition(network, eqx.is_inexact_array)
    params_mean = jax.tree_util.tree_map(jnp.mean, params)
    for k, v in jax.tree_util.tree_leaves_with_path(params_mean):
        print(k, v)


def visualize(
    key: chex.PRNGKey,
    env: Env,
    network: NormalPPONet,
    n_steps: int,
    videopath: Path | None,
    headless: bool,
    figsize: tuple[float, float],
) -> None:
    keys = jax.random.split(key, n_steps + 1)
    state, ts = env.reset(keys[0])
    obs = ts.obs
    backend = "headless" if headless else "pyglet"
    visualizer = env.visualizer(state, figsize=figsize, backend=backend)
    if videopath is not None:
        visualizer = SaveVideoWrapper(visualizer, videopath)

    # Returns action for debugging
    @eqx.filter_jit
    def step(key: chex.PRNGKey, state: State, obs: Obs) -> tuple[State, Obs, jax.Array]:
        net_out = vmap_apply(network, obs.as_array())
        actions = env.act_space.sigmoid_scale(net_out.policy().sample(seed=key))
        next_state, timestep = env.step(state, actions)
        return next_state, timestep.obs, actions

    for key in keys[1:]:
        state, obs, act = step(key, state, obs)
        # print(f"Act: {act[0]}")
        del act
        visualizer.render(state.physics)  # type: ignore
        visualizer.show()


def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    prng_key: jax.Array,
    n_rollout_steps: int,
    action_reward_coef: float,
    poison_reward_coef: float,
) -> tuple[State, Rollout, Obs, jax.Array]:
    max_norm = jnp.sqrt(jnp.sum(env.act_space.high**2, axis=-1))
    food_coef = jnp.array([[1.0, poison_reward_coef]])

    def normalize_action(action: jax.Array) -> jax.Array:
        scaled = env.act_space.sigmoid_scale(action)
        norm = jnp.sqrt(jnp.sum(scaled**2, axis=-1, keepdims=True))
        return norm / max_norm

    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, Obs], Rollout]:
        state_t, obs_t = carried
        obs_t_array = obs_t.as_array()
        net_out = vmap_apply(network, obs_t_array)
        actions = net_out.policy().sample(seed=key)
        state_t1, timestep = env.step(state_t, env.act_space.sigmoid_scale(actions))
        n_ate = timestep.info["n_ate_food"]
        act_rewards = action_reward_coef * normalize_action(actions)
        if n_ate.shape[1] == 2:
            rewards = jnp.sum(n_ate * food_coef, axis=1, keepdims=True) - act_rewards
        else:
            rewards = n_ate - act_rewards
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
    next_value = vmap_value(network, obs.as_array())
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
    reset: jax.Array,
    action_reward_coef: float,
    poison_reward_coef: float,
    entropy_weight: float,
) -> tuple[State, Obs, jax.Array, optax.OptState, NormalPPONet]:
    keys = jax.random.split(prng_key, N_MAX_AGENTS + 1)
    env_state, rollout, obs, next_value = exec_rollout(
        state,
        initial_obs,
        env,
        network,
        keys[0],
        n_rollout_steps,
        action_reward_coef,
        poison_reward_coef,
    )
    rollout = rollout.replace(terminations=rollout.terminations.at[-1].set(reset))
    batch = vmap_batch(rollout, next_value, gamma, gae_lambda)
    opt_state, pponet = vmap_update(
        batch,
        network,
        adam_update,
        opt_state,
        keys[1:],
        minibatch_size,
        n_optim_epochs,
        0.2,
        entropy_weight,
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
    action_reward_coef: float,
    poison_reward_coef: float,
    entropy_weight: float,
    figsize: tuple[float, float],
    reset_interval: int | None = None,
    debug_vis: bool = False,
) -> tuple[NormalPPONet, NDArray]:
    key, net_key, reset_key = jax.random.split(key, 3)
    obs_space = env.obs_space.flatten()
    input_size = int(np.prod(obs_space.shape))
    act_size = int(np.prod(env.act_space.shape))
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
    rewards = None
    keys = jax.random.split(key, n_loop)
    if debug_vis:
        visualizer = env.visualizer(env_state, figsize=figsize)
    else:
        visualizer = None
    for i, key in enumerate(keys):
        reset = reset_interval is not None and (i + 1) % reset_interval
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
            jnp.array(reset),
            action_reward_coef,
            poison_reward_coef,
            entropy_weight,
        )
        ri = np.array(jnp.sum(rewards_i, axis=0))
        if rewards is None:
            rewards = ri
        else:
            rewards += ri
        if visualizer is not None:
            visualizer.render(env_state.physics)  # type: ignore
            visualizer.show()
            for i in range(ri.shape[1]):
                print(f"Rewards for {i + 1}: {ri[:n_agents, i]}")
        if reset:
            env_state, timestep = env.reset(key)
            obs = timestep.obs
        # weight_summary(pponet)
    assert rewards is not None
    for i in range(rewards.shape[1]):
        print(f"Rewards for {i + 1}: {rewards[:n_agents, i]}")
    return pponet, rewards  # type: ignore


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    modelpath: Path = Path("trained.eqx"),
    seed: int = 1,
    n_agents: int = 2,
    adam_lr: float = 3e-4,
    adam_eps: float = 1e-7,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    n_optim_epochs: int = 10,
    minibatch_size: int = 256,
    n_rollout_steps: int = 1024,
    n_total_steps: int = 1024 * 1000,
    action_reward_coef: float = 1e-3,
    poison_reward_coef: float = -1.0,
    entropy_weight: float = 0.001,
    cfconfig_path: Path = PROJECT_ROOT / "config/env/20231214-square.toml",
    env_override: str = "",
    reset_interval: int | None = None,
    savelog_path: Path | None = None,
    xlim: float | None = None,
    ylim: float | None = None,
    debug_vis: bool = False,
) -> None:
    # Load config
    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())
    # Apply overrides
    cfconfig.apply_override(env_override)
    cfconfig.n_initial_agents = n_agents
    cfconfig.n_max_agents = N_MAX_AGENTS
    env = make("CircleForaging-v0", **dataclasses.asdict(cfconfig))
    xsize = cfconfig.xlim[1] * 2 if xlim is None else xlim
    ysize = cfconfig.ylim[1] * 2 if ylim is None else ylim
    network, rewards = run_training(
        key=jax.random.PRNGKey(seed),
        n_agents=n_agents,
        env=env,
        adam=optax.adam(adam_lr, eps=adam_eps),
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_optim_epochs=n_optim_epochs,
        minibatch_size=minibatch_size,
        n_rollout_steps=n_rollout_steps,
        n_total_steps=n_total_steps,
        action_reward_coef=action_reward_coef,
        poison_reward_coef=poison_reward_coef,
        entropy_weight=entropy_weight,
        figsize=(xsize, ysize),
        reset_interval=reset_interval,
        debug_vis=debug_vis,
    )
    eqx.tree_serialise_leaves(modelpath, network)
    if savelog_path is not None:
        np.savez(savelog_path, rewards)


@app.command()
def vis(
    modelpath: Path = Path("trained.eqx"),
    n_agents: int = 2,
    n_total_steps: int = 1000,
    cfconfig_path: Path = PROJECT_ROOT / "config/env/20231214-square.toml",
    seed: int = 1,
    videopath: Path | None = None,
    xlim: float | None = None,
    ylim: float | None = None,
    env_override: str = "",
    headless: bool = False,
) -> None:
    # Load config
    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())
    # Apply overrides
    cfconfig.apply_override(env_override)
    cfconfig.n_initial_agents = n_agents
    cfconfig.n_max_agents = N_MAX_AGENTS
    env = make("CircleForaging-v0", **dataclasses.asdict(cfconfig))
    obs_space = env.obs_space.flatten()
    input_size = int(np.prod(obs_space.shape))
    act_size = int(np.prod(env.act_space.shape))
    net_key, eval_key = jax.random.split(jax.random.PRNGKey(seed))
    pponet = vmap_net(
        input_size,
        64,
        act_size,
        jax.random.split(net_key, N_MAX_AGENTS),
    )
    pponet = eqx.tree_deserialise_leaves(modelpath, pponet)
    xsize = cfconfig.xlim[1] * 2 if xlim is None else xlim
    ysize = cfconfig.ylim[1] * 2 if ylim is None else ylim
    visualize(
        key=eval_key,
        env=env,
        network=pponet,
        n_steps=n_total_steps,
        videopath=videopath,
        headless=headless,
        figsize=(xsize, ysize),
    )


if __name__ == "__main__":
    app()
