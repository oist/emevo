"""Example of using circle foraging environment"""
import dataclasses
import enum
from pathlib import Path
from typing import Protocol

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from serde import toml

from emevo import Env
from emevo import birth_and_death as bd
from emevo import make
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.exp_utils import BDConfig, CfConfig, Log
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


class RewardFn(Protocol):
    def __call__(self, collision: jax.Array, action: jax.Array) -> jax.Array:
        ...


class LinearReward(eqx.Module):
    weight: jax.Array

    def __init__(self, key: chex.PRNGKey, n_agents: int) -> None:
        self.weight = jax.random.normal(key, (n_agents, 4))

    def __call__(self, collision: jax.Array, action: jax.Array) -> jax.Array:
        action_norm = jnp.sqrt(jnp.sum(action**2, axis=-1, keepdims=True))
        input_ = jnp.concatenate((collision, action_norm), axis=1)
        return jax.vmap(jnp.dot)(input_, self.weight)


def evolve_rewards(old, method, parents: jax.Array):
    pass


class RewardKind(str, enum.Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"


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
    reward_fn: LinearReward,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Rollout, Log, Obs, jax.Array]:
    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, Obs], tuple[Rollout, Log]]:
        act_key, hazard_key, birth_key = jax.random.split(key, 3)
        state_t, obs_t = carried
        obs_t_array = obs_t.as_array()
        net_out = vmap_apply(network, obs_t_array)
        actions = net_out.policy().sample(seed=act_key)
        state_t1, timestep = env.step(state_t, env.act_space.sigmoid_scale(actions))
        rewards = reward_fn(obs_t.collision, actions).reshape(-1, 1)
        rollout = Rollout(
            observations=obs_t_array,
            actions=actions,
            rewards=rewards,
            terminations=jnp.zeros_like(rewards),
            values=net_out.value,
            means=net_out.mean,
            logstds=net_out.logstd,
        )
        # Birth and death
        death_prob = hazard_fn(state_t1.status.age, state_t1.status.energy)
        dead = jax.random.bernoulli(hazard_key, p=death_prob)
        state_t1d = env.deactivate(state_t1, dead)
        birth_prob = birth_fn(state_t1d.status.age, state_t1d.status.energy)
        possible_parents = jnp.logical_and(
            jnp.logical_and(jnp.logical_not(dead), state.profile.is_active()),
            jax.random.bernoulli(birth_key, p=birth_prob),
        )
        state_t1db, parents = env.activate(state_t1d, possible_parents)
        log = Log(
            parents=parents,
            rewards=rewards.ravel(),
            age=state_t1db.status.age,
            energy=state_t1db.status.energy,
            birthtime=state_t1db.profile.birthtime,
            generation=state_t1db.profile.generation,
            unique_id=state_t1db.profile.unique_id,
        )
        return (state_t1db, timestep.obs), (rollout, log)

    (state, obs), (rollout, log) = jax.lax.scan(
        step_rollout,
        (state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    next_value = vmap_value(network, obs.as_array())
    return state, rollout, log, obs, next_value


@eqx.filter_jit
def epoch(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    reward_fn: LinearReward,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    prng_key: jax.Array,
    n_rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    adam_update: optax.TransformUpdateFn,
    opt_state: optax.OptState,
    minibatch_size: int,
    n_optim_epochs: int,
) -> tuple[State, Obs, Log, optax.OptState, NormalPPONet]:
    keys = jax.random.split(prng_key, env.n_max_agents + 1)
    env_state, rollout, log, obs, next_value = exec_rollout(
        state,
        initial_obs,
        env,
        network,
        reward_fn,
        hazard_fn,
        birth_fn,
        keys[0],
        n_rollout_steps,
    )
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
        0.0,
    )
    return env_state, obs, log, opt_state, pponet


def run_evolution(
    key: jax.Array,
    env: Env,
    adam: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    n_optim_epochs: int,
    minibatch_size: int,
    n_rollout_steps: int,
    n_total_steps: int,
    reward_fn: LinearReward,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    logdir: Path,
    log_interval: int,
    xmax: float,
    ymax: float,
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
        jax.random.split(net_key, env.n_max_agents),
    )
    adam_init, adam_update = adam
    opt_state = jax.vmap(adam_init)(eqx.filter(pponet, eqx.is_array))
    env_state, timestep = env.reset(reset_key)
    obs = timestep.obs

    n_loop = n_total_steps // n_rollout_steps
    keys = jax.random.split(key, n_loop)
    if debug_vis:
        visualizer = env.visualizer(env_state, figsize=(xmax * 2, ymax * 2))
    else:
        visualizer = None

    log_list = []

    def write_log(index: int) -> None:
        log = jax.tree_map(
            lambda *args: np.array(jnp.concatenate(args, axis=0)),
            *log_list,
        )
        log_dict = dataclasses.asdict(log)
        table = pa.Table.from_pydict(log_dict)
        pq.write_table(
            table,
            logdir.joinpath(f"log-{index}.parquet"),
            compression="zstd",
        )
        log_list.clear()

    for i, key in enumerate(keys):
        env_state, obs, log, opt_state, pponet = epoch(
            env_state,
            obs,
            env,
            pponet,
            reward_fn,
            hazard_fn,
            birth_fn,
            key,
            n_rollout_steps,
            gamma,
            gae_lambda,
            adam_update,
            opt_state,
            minibatch_size,
            n_optim_epochs,
        )
        if visualizer is not None:
            visualizer.render(env_state)
            visualizer.show()

        # Extinct?
        n_active = jnp.sum(env_state.profile.is_active())
        if n_active == 0:
            print(f"Extinct after {i + 1} epochs")
            return pponet

        filtered_log = log.with_step(i * n_rollout_steps).filter()

        log_list.append(filtered_log)
        if (i + 1) % log_interval == 0:
            index = (i + 1) // log_interval
            write_log(index)

    return pponet


app = typer.Typer(pretty_exceptions_show_locals=False)
here = Path(__file__).parent


@app.command()
def evolve(
    seed: int = 1,
    n_agents: int = 20,
    adam_lr: float = 3e-4,
    adam_eps: float = 1e-7,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    n_optim_epochs: int = 10,
    minibatch_size: int = 128,
    n_rollout_steps: int = 1024,
    n_total_steps: int = 1024 * 10000,
    cfconfig_path: Path = here.joinpath("../config/env/20231214-square.toml"),
    bdconfig_path: Path = here.joinpath("../config/bd/20230530-a035-e020.toml"),
    reward_fn: RewardKind = RewardKind.LINEAR,
    logdir: Path = Path("./log"),
    log_interval: int = 100,
    debug_vis: bool = False,
) -> None:
    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())
    with bdconfig_path.open("r") as f:
        bdconfig = toml.from_toml(BDConfig, f.read())

    # Override config
    cfconfig.n_initial_agents = n_agents
    env = make("CircleForaging-v0", **dataclasses.asdict(cfconfig))
    birth_fn, hazard_fn = bdconfig.load_models()
    key, reward_key = jax.random.split(jax.random.PRNGKey(seed))
    if reward_fn == RewardKind.LINEAR:
        reward_fn_instance = LinearReward(reward_key, cfconfig.n_max_agents)
    elif reward_fn == RewardKind.SIGMOID:
        assert False, "Unimplemented"
    else:
        raise ValueError(f"Invalid reward_fn {reward_fn}")
    network = run_evolution(
        key,
        env,
        optax.adam(adam_lr, eps=adam_eps),
        gamma,
        gae_lambda,
        n_optim_epochs,
        minibatch_size,
        n_rollout_steps,
        n_total_steps,
        reward_fn_instance,
        hazard_fn,
        birth_fn,
        logdir,
        log_interval,
        cfconfig.xlim[1],
        cfconfig.ylim[1],
        debug_vis,
    )
    # eqx.tree_serialise_leaves(modelpath, network)


@app.command()
def vis() -> None:
    assert False, "Unimplemented"


if __name__ == "__main__":
    app()
