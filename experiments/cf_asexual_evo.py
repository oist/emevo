"""Example of using circle foraging environment"""

from pathlib import Path
from typing import Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer

from emevo import Env, make
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.rl.ppo_normal import NormalPPONet
from emevo.rl.ppo_normal import Rollout as OriginalRollout
from emevo.rl.ppo_normal import (
    vmap_apply,
    vmap_batch,
    vmap_net,
    vmap_update,
    vmap_value,
)
from emevo.visualizer import SaveVideoWrapper

N_MAX_AGENTS: int = 10


@chex.dataclass
class Rollout(OriginalRollout):
    collision: jax.Array


class LinearReward(eqx.Module):
    weight: jax.Array
    max_action_norm: float

    def __init__(self, max_action_norm: float, key: chex.PRNGKey) -> None:
        self.weight = jax.random.normal(key, (1, 4))
        self.max_action_norm = max_action_norm

    def __call__(self, collision: jax.Array, action: jax.Array) -> jax.Array:
        action_norm = jnp.sqrt(jnp.sum(action**2, axis=-1, keepdims=True))
        return jnp.concatenate((collision, action_norm), axis=1) @ self.weight.T


def weight_summary(network):
    params, _ = eqx.partition(network, eqx.is_inexact_array)
    params_mean = jax.tree_map(jnp.mean, params)
    for k, v in jax.tree_util.tree_leaves_with_path(params_mean):
        print(k, v)


def visualize(
    key: chex.PRNGKey,
    env: Env,
    network: NormalPPONet,
    n_steps: int,
    videopath: Path | None,
    headless: bool,
) -> None:
    keys = jax.random.split(key, n_steps + 1)
    state, ts = env.reset(keys[0])
    obs = ts.obs
    backend = "headless" if headless else "pyglet"
    visualizer = env.visualizer(state, figsize=(640.0, 640.0), backend=backend)
    if videopath is not None:
        visualizer = SaveVideoWrapper(visualizer, videopath, fps=60)

    # Returns action for debugging
    @eqx.filter_jit
    def step(key: chex.PRNGKey, state: State, obs: Obs) -> tuple[State, Obs, jax.Array]:
        net_out = vmap_apply(network, obs.as_array())
        actions = net_out.policy().sample(seed=key)
        next_state, timestep = env.step(state, env.act_space.sigmoid_scale(actions))
        return next_state, timestep.obs, actions

    for key in keys[1:]:
        state, obs, act = step(key, state, obs)
        # print(f"Act: {act[0]}")
        visualizer.render(state)
        visualizer.show()


def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    reward_fn: RewardFn,
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
        state_t1, timestep = env.step(state_t, env.act_space.sigmoid_scale(actions))
        rewards = reward_fn()
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
    rollout = rollout.replace(terminations=rollout.terminations.at[-1].set(reset))
    batch = vmap_batch(rollout, next_value, gamma, gae_lambda)
    output = vmap_apply(network, obs.as_array())
    opt_state, pponet = vmap_update(
        batch,
        network,
        adam_update,
        opt_state,
        keys[1:],
        minibatch_size,
        n_optim_epochs,
        0.2,
        0.0,
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
    reset_interval: int | None = None,
    debug_vis: bool = False,
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
    if debug_vis:
        visualizer = env.visualizer(env_state, figsize=(640.0, 640.0))
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
        )
        ri = jnp.sum(jnp.squeeze(rewards_i, axis=-1), axis=0)
        rewards = rewards + ri
        if visualizer is not None:
            visualizer.render(env_state)
            visualizer.show()
            print(f"Rewards: {[x.item() for x in ri[: n_agents]]}")
        if reset:
            env_state, timestep = env.reset(key)
            obs = timestep.obs
        # weight_summary(pponet)
    print(f"Sum of rewards {[x.item() for x in rewards[: n_agents]]}")
    return pponet


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    modelpath: Path = Path("trained.eqx"),
    seed: int = 1,
    n_agents: int = 2,
    n_foods: int = 10,
    obstacles: str = "none",
    adam_lr: float = 3e-4,
    adam_eps: float = 1e-7,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    n_optim_epochs: int = 10,
    minibatch_size: int = 128,
    n_rollout_steps: int = 1024,
    n_total_steps: int = 1024 * 1000,
    food_loc_fn: str = "gaussian",
    env_shape: str = "circle",
    reset_interval: Optional[int] = None,
    xlim: int = 200,
    ylim: int = 200,
    linear_damping: float = 0.8,
    angular_damping: float = 0.6,
    max_force: float = 40.0,
    min_force: float = -20.0,
    debug_vis: bool = False,
) -> None:
    assert n_agents < N_MAX_AGENTS
    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=n_agents,
        food_num_fn=("constant", n_foods),
        food_loc_fn=food_loc_fn,
        agent_loc_fn="gaussian",
        foodloc_interval=20,
        obstacles=obstacles,
        xlim=(0.0, float(xlim)),
        ylim=(0.0, float(ylim)),
        env_radius=min(xlim, ylim) * 0.5,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        max_force=max_force,
        min_force=min_force,
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
        reset_interval,
        debug_vis,
    )
    eqx.tree_serialise_leaves(modelpath, network)


@app.command()
def vis() -> None:
    assert False, "Unimplemented"


if __name__ == "__main__":
    app()
