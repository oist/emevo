"""Asexual reward evolution with Circle Foraging"""

import dataclasses
import json
from pathlib import Path
from typing import cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from phyjax2d import Position
from serde import serde, toml

from emevo import Env
from emevo import birth_and_death as bd
from emevo import genetic_ops as gops
from emevo import make
from emevo import reward_fn as rfn
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.eqx_utils import get_slice
from emevo.eqx_utils import where as eqx_where
from emevo.exp_utils import (
    BDConfig,
    CfConfig,
    FoodLog,
    GopsConfig,
    Log,
    Logger,
    LogMode,
    SavedPhysicsState,
    SavedProfile,
    is_cuda_ready,
)
from emevo.reward_extractor import SensorActFoodExtractor as RewardExtractor
from emevo.rl import ppo_normal as ppo
from emevo.visualizer import SaveVideoWrapper

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CFCONFIG = PROJECT_ROOT / "config/env/20250207-obstacle.toml"
SENSOR_NAMES = ["agent_sensor", "obs_sensor"]
N_SENSOR_REWARDS = 2


def serialize_weight(w: jax.Array) -> dict[str, jax.Array]:
    wd = w.shape[0]
    n_food_rewards = wd - N_SENSOR_REWARDS - 1
    rd = {f"food_{i + 1}": rfn.slice_last(w, i) for i in range(n_food_rewards)}
    rd["action"] = rfn.slice_last(w, n_food_rewards)
    for i, sensor_name in enumerate(SENSOR_NAMES):
        rd[sensor_name] = rfn.slice_last(w, i + n_food_rewards + 1)
    return rd


def get_mean_sensor_obs(sensor_obs: jax.Array) -> jax.Array:
    # E.g., sensor with predator: (N_agents, N_sensors, N_obj)
    used_sensor_obs = sensor_obs[:, :, :N_SENSOR_REWARDS]
    clipped_sensor_obs = jnp.clip(used_sensor_obs, min=0.0)
    return jnp.mean(clipped_sensor_obs, axis=1)  # (N_agents, N_obj)


def get_closest_sensor_obs(sensor_obs: jax.Array) -> jax.Array:
    used_sensor_obs = sensor_obs[:, :, :N_SENSOR_REWARDS]
    clipped_sensor_obs = jnp.clip(used_sensor_obs, min=0.0)
    return jnp.max(clipped_sensor_obs, axis=1)  # (N_agents, N_obj)


@serde
@dataclasses.dataclass
class CfConfigWithObstacle(CfConfig):
    n_obstacles: int = 4
    obstacle_damage: float = 10.0
    obstacle_size: float = 20.0
    n_hol_blocks: int = 10
    n_vert_blocks: int = 10
    obstacle_schedule: list[tuple[int, int]] = dataclasses.field(default_factory=list)


def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: ppo.NormalPPONet,
    reward_fn: rfn.RewardFn,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    prng_key: jax.Array,
    n_rollout_steps: int,
    sensor_agg_type: str = "mean",
) -> tuple[State, ppo.Rollout, Log, FoodLog, SavedPhysicsState, Obs, jax.Array]:
    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, Obs], tuple[ppo.Rollout, Log, FoodLog, SavedPhysicsState]]:
        act_key, hazard_key, birth_key = jax.random.split(key, 3)
        state_t, obs_t = carried
        obs_t_array = obs_t.as_array()
        net_out = ppo.vmap_apply(network, obs_t_array)
        actions = net_out.policy().sample(seed=act_key)
        state_t1, timestep = env.step(
            state_t,
            env.act_space.sigmoid_scale(actions),  # type: ignore
        )
        obs_t1 = timestep.obs
        if sensor_agg_type == "mean":
            agg_sensor_obs = get_mean_sensor_obs(obs_t1.sensor)
        else:
            agg_sensor_obs = get_closest_sensor_obs(obs_t1.sensor)
        rewards = jnp.expand_dims(
            reward_fn(timestep.info["n_ate_food"], actions, agg_sensor_obs),
            axis=1,
        )
        rollout = ppo.Rollout(
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
        dead_nonzero = jax.random.bernoulli(hazard_key, p=death_prob)
        dead = jnp.where(
            # If the agent's energy is lower than 0, it should immediately die
            state_t1.status.energy < 0.0,
            jnp.ones_like(dead_nonzero),
            dead_nonzero,
        )
        state_t1d = env.deactivate(state_t1, dead)
        birth_prob = birth_fn(state_t1d.status.age, state_t1d.status.energy)
        possible_parents = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(dead),
                state.unique_id.is_active(),  # type: ignore
            ),
            jax.random.bernoulli(birth_key, p=birth_prob),
        )
        state_t1db, parents = env.activate(state_t1d, possible_parents)
        log = Log(
            dead=jnp.where(dead, state_t.unique_id.unique_id, -1),  # type: ignore
            n_got_food=timestep.info["n_ate_food"],
            action_magnitude=actions,
            energy_gain=timestep.info["energy_gain"],
            consumed_energy=timestep.info["energy_consumption"],
            energy=state_t1db.status.energy,
            parents=parents,
            rewards=rewards.ravel(),
            unique_id=state_t1db.unique_id.unique_id,
            additional_fields={
                "obstacle_damage": timestep.info["obstacle_damage"],
            },
        )
        foodlog = FoodLog(
            eaten=timestep.info["n_food_eaten"],
            regenerated=timestep.info["n_food_regenerated"],
        )
        phys = state_t.physics  # type: ignore
        phys_state = SavedPhysicsState(
            circle_axy=phys.circle.p.into_axy(),
            static_circle_axy=phys.static_circle.p.into_axy(),
            circle_is_active=phys.circle.is_active,
            static_circle_is_active=phys.static_circle.is_active,
            static_circle_label=phys.static_circle.label,
        )
        return (state_t1db, obs_t1), (rollout, log, foodlog, phys_state)

    (state, obs), (rollout, log, foodlog, phys_state) = jax.lax.scan(
        step_rollout,
        (state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    next_value = ppo.vmap_value(network, obs.as_array())
    return state, rollout, log, foodlog, phys_state, obs, next_value


@eqx.filter_jit
def epoch(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: ppo.NormalPPONet,
    reward_fn: rfn.RewardFn,
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
    entropy_weight: float,
    sensor_agg_type: str = "mean",
) -> tuple[
    State, Obs, Log, FoodLog, SavedPhysicsState, optax.OptState, ppo.NormalPPONet
]:
    is_updatable = state.unique_id.is_active()

    keys = jax.random.split(prng_key, env.n_max_agents + 1)
    env_state, rollout, log, foodlog, phys_state, obs, next_value = exec_rollout(
        state,
        initial_obs,
        env,
        network,
        reward_fn,
        hazard_fn,
        birth_fn,
        keys[0],
        n_rollout_steps,
        sensor_agg_type=sensor_agg_type,
    )
    batch = ppo.vmap_batch(rollout, next_value, gamma, gae_lambda)
    opt_state, updated_network = ppo.vmap_update(
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
    # Filter the updates of inactive network
    filtered_network = eqx_where(is_updatable, updated_network, network)
    return env_state, obs, log, foodlog, phys_state, opt_state, filtered_network


def run_evolution(
    *,
    key: jax.Array,
    env: Env,
    adam: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    n_optim_epochs: int,
    minibatch_size: int,
    n_rollout_steps: int,
    n_total_steps: int,
    entropy_weight: float,
    reward_fn: rfn.RewardFn,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    mutation: gops.Mutation,
    xmax: float,
    ymax: float,
    logger: Logger,
    save_interval: int,
    debug_vis: bool,
    debug_vis_scale: float,
    debug_print: bool,
    headless: bool,
    sensor_agg_type: str = "mean",
) -> None:
    key, net_key, reset_key = jax.random.split(key, 3)
    obs_space = env.obs_space.flatten()
    input_size = int(np.prod(obs_space.shape))
    act_size = int(np.prod(env.act_space.shape))

    def initialize_net(key: chex.PRNGKey) -> ppo.NormalPPONet:
        return ppo.vmap_net(
            input_size,
            64,
            act_size,
            jax.random.split(key, env.n_max_agents),
        )

    pponet = initialize_net(net_key)
    adam_init, adam_update = adam

    @eqx.filter_jit
    def initialize_opt_state(net: eqx.Module) -> optax.OptState:
        return jax.vmap(adam_init)(eqx.filter(net, eqx.is_array))

    @eqx.filter_jit
    def replace_net(
        key: chex.PRNGKey,
        flag: jax.Array,
        pponet: ppo.NormalPPONet,
        opt_state: optax.OptState,
    ) -> tuple[ppo.NormalPPONet, optax.OptState]:
        initialized = initialize_net(key)
        pponet = eqx_where(flag, initialized, pponet)
        opt_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(
                jnp.expand_dims(flag, tuple(range(1, a.ndim))),
                b,
                a,
            ),
            opt_state,
            initialize_opt_state(pponet),
        )
        return pponet, opt_state

    opt_state = initialize_opt_state(pponet)
    env_state, timestep = env.reset(reset_key)
    np.savez(
        logger.logdir / "obstacles.npz",
        obstacle_axy=np.array(env_state.physics.static_triangle.p.into_axy()),
    )
    obs = timestep.obs

    if debug_vis:
        visualizer = env.visualizer(
            env_state,
            figsize=(xmax * debug_vis_scale, ymax * debug_vis_scale),
            backend="headless" if headless else "pyglet",
        )
    else:
        visualizer = None

    for i, is_active in enumerate(env_state.unique_id.is_active()):
        if is_active:
            logger.reward_fn_dict[i + 1] = get_slice(reward_fn, i)
            logger.profile_dict[i + 1] = SavedProfile(0, 0, i + 1)

    all_keys = jax.random.split(key, n_total_steps // n_rollout_steps)
    del key  # Don't reuse this key!
    for i, key_i in enumerate(all_keys):
        epoch_key, mutation_key, init_key = jax.random.split(key_i, 3)
        old_state = env_state
        # Use `with jax.disable_jit():` here for debugging
        env_state, obs, log, foodlog, phys_state, opt_state, pponet = epoch(
            env_state,
            obs,
            env,
            pponet,
            reward_fn,
            hazard_fn,
            birth_fn,
            epoch_key,
            n_rollout_steps,
            gamma,
            gae_lambda,
            adam_update,
            opt_state,
            minibatch_size,
            n_optim_epochs,
            entropy_weight,
            sensor_agg_type=sensor_agg_type,
        )

        if visualizer is not None:
            visualizer.render(env_state.physics)  # type: ignore
            visualizer.show()
        if debug_print:
            is_active = env_state.unique_id.is_active()
            popl = int(jnp.sum(is_active))
            avg_e = float(jnp.mean(env_state.status.energy[is_active]))
            if popl > 0:
                print(f"Population: {popl} Avg. Energy: {avg_e}")

        # Extinct?
        n_active = jnp.sum(env_state.unique_id.is_active())  # type: ignore
        if n_active == 0:
            print(f"Extinct after {i + 1} epochs")
            break

        # Save dead agents
        log_with_step = log.with_step(i * n_rollout_steps)
        log_death = log_with_step.filter_death()
        ages = old_state.status.age[log_death.slots]
        logger.save_agents(
            pponet,
            log_death.log.dead,
            log_death.slots,
            ages + log_death.step - i * n_rollout_steps,
        )
        # Save alive agents
        saved = jnp.logical_and(
            env_state.status.age > 0,
            ((env_state.status.age // n_rollout_steps) % save_interval) == 0,
        )
        (saved_slots,) = jnp.nonzero(saved)
        logger.save_agents(
            pponet,
            env_state.unique_id.unique_id[saved_slots],
            saved_slots,
            env_state.status.age[saved_slots],
            prefix="intermediate",
        )
        # Initialize network and adam state for new agents
        log_birth = log_with_step.filter_birth()
        is_new = jnp.zeros(env.n_max_agents, dtype=bool).at[log_birth.slots].set(True)
        if jnp.any(is_new):
            pponet, opt_state = replace_net(init_key, is_new, pponet, opt_state)
        # Mutation
        reward_fn = rfn.mutate_reward_fn(
            mutation_key,
            logger.reward_fn_dict,
            reward_fn,
            mutation,
            log_birth.log.parents,
            log_birth.log.unique_id,
            log_birth.slots,
        )
        # Update profile
        for step, uid, parent in zip(
            log_birth.step,
            log_birth.log.unique_id,
            log_birth.log.parents,
        ):
            ui = uid.item()
            logger.profile_dict[ui] = SavedProfile(step.item(), parent.item(), ui)

        # Push log and physics state
        logger.push_foodlog(foodlog)
        logger.push_log(log_with_step.filter_active())
        logger.push_physstate(phys_state)

    # Save logs before exiting
    logger.finalize()
    is_active = env_state.unique_id.is_active()
    logger.save_agents(
        pponet,
        env_state.unique_id.unique_id[is_active],
        jnp.arange(len(is_active))[is_active],
        env_state.status.age[is_active],
    )


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def evolve(
    seed: int = 1,
    adam_lr: float = 3e-4,
    adam_eps: float = 1e-7,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    n_optim_epochs: int = 10,
    minibatch_size: int = 256,
    n_rollout_steps: int = 1024,
    n_total_steps: int = 1024 * 10000,
    act_reward_coef: float = 0.01,
    sensor_reward_coef: float = 0.1,
    entropy_weight: float = 0.001,
    cfconfig_path: Path = DEFAULT_CFCONFIG,
    bdconfig_path: Path = PROJECT_ROOT / "config/bd/20240318-mild-slope.toml",
    gopsconfig_path: Path = PROJECT_ROOT / "config/gops/20240326-cauthy-002.toml",
    min_age_for_save: int = 0,
    save_interval: int = 100000000,  # No saving by default
    env_override: str = "",
    birth_override: str = "",
    hazard_override: str = "",
    gops_params_override: str = "",
    logdir: Path = Path("./log"),
    log_mode: LogMode = LogMode.REWARD_LOG_STATE,
    log_interval: int = 1000,
    savestate_interval: int = 1000,
    debug_vis: bool = False,
    debug_vis_scale: float = 1.0,
    debug_print: bool = False,
    headless: bool = False,
    sensor_agg_type: str = "max",
    force_gpu: bool = True,
) -> None:
    if force_gpu and not is_cuda_ready():
        raise RuntimeError("Detected some problem in CUDA!")

    # Load config
    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfigWithObstacle, f.read())
    with bdconfig_path.open("r") as f:
        bdconfig = toml.from_toml(BDConfig, f.read())
    with gopsconfig_path.open("r") as f:
        gopsconfig = toml.from_toml(GopsConfig, f.read())

    # Apply overrides
    cfconfig.apply_override(env_override)
    bdconfig.apply_birth_override(birth_override)
    bdconfig.apply_hazard_override(hazard_override)
    gopsconfig.apply_params_override(gops_params_override)

    # Load models
    birth_fn, hazard_fn = bdconfig.load_models()
    mutation = gopsconfig.load_model()
    # Make env
    env = make("CircleForaging-v3", **dataclasses.asdict(cfconfig))
    key, reward_key = jax.random.split(jax.random.PRNGKey(seed))

    reward_extracor = RewardExtractor(
        act_space=env.act_space,  # type: ignore
        act_coef=act_reward_coef,
        sensor_coef=sensor_reward_coef,
    )
    if cfconfig.observe_food_label:
        n_food_obs = cfconfig.n_food_sources
    else:
        n_food_obs = 1
    reward_fn_instance = rfn.LinearReward(
        key=reward_key,
        n_agents=cfconfig.n_max_agents,
        n_weights=1 + n_food_obs + N_SENSOR_REWARDS,
        std=gopsconfig.init_std,
        mean=gopsconfig.init_mean,
        extractor=reward_extracor.extract,
        serializer=serialize_weight,
        **gopsconfig.init_kwargs,
    )

    logger = Logger(
        logdir=logdir,
        mode=log_mode,
        log_interval=log_interval,
        savestate_interval=savestate_interval,
        min_age_for_save=min_age_for_save,
    )
    run_evolution(
        key=key,
        env=env,
        adam=optax.adam(adam_lr, eps=adam_eps),
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_optim_epochs=n_optim_epochs,
        minibatch_size=minibatch_size,
        n_rollout_steps=n_rollout_steps,
        n_total_steps=n_total_steps,
        entropy_weight=entropy_weight,
        reward_fn=reward_fn_instance,
        hazard_fn=hazard_fn,
        birth_fn=birth_fn,
        mutation=cast(gops.Mutation, mutation),
        xmax=cfconfig.xlim[1],
        ymax=cfconfig.ylim[1],
        logger=logger,
        save_interval=save_interval,
        debug_vis=debug_vis,
        debug_vis_scale=debug_vis_scale,
        headless=headless,
        debug_print=debug_vis or debug_print,
        sensor_agg_type=sensor_agg_type,
    )


@app.command()
def replay(
    physstate_path: Path,
    backend: str = "pyglet",  # Use "headless" for headless rendering
    videopath: Path | None = None,
    start: int = 0,
    end: int | None = None,
    cfconfig_path: Path = DEFAULT_CFCONFIG,
    env_override: str = "",
    scale: float = 1.0,
    force_cpu: bool = False,
) -> None:
    if force_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])

    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfigWithObstacle, f.read())
    # For speedup
    cfconfig.n_initial_agents = 1
    cfconfig.apply_override(env_override)
    phys_state = SavedPhysicsState.load(physstate_path)
    env = make("CircleForaging-v3", **dataclasses.asdict(cfconfig))
    env_state, _ = env.reset(jax.random.PRNGKey(0))
    npzfile = np.load(physstate_path.parent / "obstacles.npz")
    obstacle_axy = jnp.array(npzfile["obstacle_axy"])
    env_state = dataclasses.replace(
        env_state,
        physics=env_state.physics.nested_replace(
            "static_triangle.p",
            Position.from_axy(obstacle_axy),
        ),
    )
    end_index = end if end is not None else phys_state.circle_axy.shape[0]
    visualizer = env.visualizer(
        env_state,
        figsize=(cfconfig.xlim[1] * scale, cfconfig.ylim[1] * scale),
        backend=backend,
    )
    if videopath is not None:
        visualizer = SaveVideoWrapper(visualizer, videopath)
    for i in range(start, end_index):
        ph = phys_state.set_by_index(i, env_state.physics)
        # Disable rendering agents
        # ph = ph.nested_replace("circle.is_active", jnp.zeros_like(ph.circle.is_active))
        env_state = dataclasses.replace(env_state, physics=ph)
        visualizer.render(env_state.physics)
        visualizer.show()
    visualizer.close()


@app.command()
def widget(
    physstate_path: Path,
    start: int = 0,
    end: int | None = None,
    cfconfig_path: Path = DEFAULT_CFCONFIG,
    log_path: Path | None = None,
    self_terminate: bool = False,
    profile_and_rewards_path: Path | None = None,
    cm_fixed_minmax: str = "",
    env_override: str = "",
    scale: float = 2.0,
    force_cpu: bool = False,
) -> None:
    from emevo.analysis.qt_widget import CFEnvReplayWidget, start_widget

    if force_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])

    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfigWithObstacle, f.read())

    # For speedup
    cfconfig.n_initial_agents = 1
    cfconfig.apply_override(env_override)
    phys_state = SavedPhysicsState.load(physstate_path)
    print("Loaded phys state")
    env = make("CircleForaging-v3", **dataclasses.asdict(cfconfig))
    npzfile = np.load(physstate_path.parent / "obstacles.npz")
    obstacle_axy = jnp.array(npzfile["obstacle_axy"])

    def initialize_env_state(state):
        return dataclasses.replace(
            state,
            physics=state.physics.nested_replace(
                "static_triangle.p",
                Position.from_axy(obstacle_axy),
            ),
        )

    end = phys_state.circle_axy.shape[0] if end is None else end
    if log_path is None:
        log_ds = None
        step_offset = 0
    else:
        import pyarrow.dataset as ds

        log_ds = ds.dataset(log_path)
        step_offset = log_ds.scanner(columns=["step"]).head(1)["step"][0].as_py()

    if profile_and_rewards_path is None:
        profile_and_rewards = None
    else:
        import pyarrow.parquet as pq

        profile_and_rewards = pq.read_table(profile_and_rewards_path)

    if len(cm_fixed_minmax) > 0:
        cm_fixed_minmax_dict = json.loads(cm_fixed_minmax)
    else:
        cm_fixed_minmax_dict = {}

    start_widget(
        CFEnvReplayWidget,
        xlim=int(cfconfig.xlim[1]),
        ylim=int(cfconfig.ylim[1]),
        env=env,
        saved_physics=phys_state,
        start=start,
        end=end,
        log_ds=log_ds,
        step_offset=step_offset,
        self_terminate=self_terminate,
        profile_and_rewards=profile_and_rewards,
        cm_fixed_minmax=cm_fixed_minmax_dict,
        scale=scale,
        initialize_env_state=initialize_env_state,
        show_obstacle_info=True,
    )


if __name__ == "__main__":
    app()
