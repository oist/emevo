"""cf_simple without learning"""

import dataclasses
from pathlib import Path
from typing import cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import typer
from serde import toml

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
from emevo.rl.ppo_normal import (
    NormalPPONet,
    vmap_apply,
    vmap_net,
)
from emevo.spaces import BoxSpace

PROJECT_ROOT = Path(__file__).parent.parent


@dataclasses.dataclass
class RewardExtractor:
    act_space: BoxSpace
    act_coef: float
    _max_norm: jax.Array = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._max_norm = jnp.sqrt(jnp.sum(self.act_space.high**2, axis=-1))

    def normalize_action(self, action: jax.Array) -> jax.Array:
        scaled = self.act_space.sigmoid_scale(action)
        norm = jnp.sqrt(jnp.sum(scaled**2, axis=-1, keepdims=True))
        return norm / self._max_norm

    def extract(
        self,
        ate_food: jax.Array,
        action: jax.Array,
        energy: jax.Array,
    ) -> jax.Array:
        del energy
        act_input = self.act_coef * self.normalize_action(action)
        return jnp.concatenate((ate_food.astype(jnp.float32), act_input), axis=1)


def serialize_weight(w: jax.Array) -> dict[str, jax.Array]:
    wd = w.shape[0]
    rd = {f"food_{i + 1}": rfn.slice_last(w, i) for i in range(wd - 1)}
    rd["action"] = rfn.slice_last(w, wd - 1)
    return rd


def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    reward_fn: rfn.RewardFn,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Log, FoodLog, SavedPhysicsState, Obs]:
    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, Obs], tuple[Log, FoodLog, SavedPhysicsState]]:
        act_key, hazard_key, birth_key = jax.random.split(key, 3)
        state_t, obs_t = carried
        obs_t_array = obs_t.as_array()
        net_out = vmap_apply(network, obs_t_array)
        actions = net_out.policy().sample(seed=act_key)
        state_t1, timestep = env.step(
            state_t,
            env.act_space.sigmoid_scale(actions),  # type: ignore
        )
        obs_t1 = timestep.obs
        energy = state_t.status.energy
        rewards = reward_fn(timestep.info["n_ate_food"], actions, energy).reshape(-1, 1)
        # Birth and death
        death_prob = hazard_fn(state_t1.status.age, state_t1.status.energy)
        dead = jax.random.bernoulli(hazard_key, p=death_prob)
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
            consumed_energy=timestep.info["energy_consumption"],
            energy=state_t1db.status.energy,
            parents=parents,
            rewards=rewards.ravel(),
            unique_id=state_t1db.unique_id.unique_id,
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
        return (state_t1db, obs_t1), (log, foodlog, phys_state)

    (state, obs), (log, foodlog, phys_state) = jax.lax.scan(
        step_rollout,
        (state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    return state, log, foodlog, phys_state, obs


@eqx.filter_jit
def epoch(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    reward_fn: rfn.RewardFn,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Obs, Log, FoodLog, SavedPhysicsState, NormalPPONet]:
    keys = jax.random.split(prng_key, env.n_max_agents + 1)
    env_state, log, foodlog, phys_state, obs = exec_rollout(
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
    return env_state, obs, log, foodlog, phys_state, network


def run_noevo(
    *,
    key: jax.Array,
    env: Env,
    n_initial_agents: int,
    n_rollout_steps: int,
    n_total_steps: int,
    reward_fn: rfn.RewardFn,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    mutation: gops.Mutation,
    xmax: float,
    ymax: float,
    logger: Logger,
    debug_vis: bool,
) -> None:
    key, net_key, reset_key = jax.random.split(key, 3)
    obs_space = env.obs_space.flatten()
    input_size = np.prod(obs_space.shape)
    act_size = np.prod(env.act_space.shape)

    def initialize_net(key: chex.PRNGKey) -> NormalPPONet:
        return vmap_net(
            input_size,
            64,
            act_size,
            jax.random.split(key, env.n_max_agents),
        )

    pponet = initialize_net(net_key)

    @eqx.filter_jit
    def replace_net(
        key: chex.PRNGKey,
        flag: jax.Array,
        pponet: NormalPPONet,
    ) -> NormalPPONet:
        initialized = initialize_net(key)
        return eqx_where(flag, initialized, pponet)

    env_state, timestep = env.reset(reset_key)
    obs = timestep.obs

    if debug_vis:
        visualizer = env.visualizer(env_state, figsize=(xmax * 2, ymax * 2))
    else:
        visualizer = None

    for i in range(n_initial_agents):
        logger.reward_fn_dict[i + 1] = get_slice(reward_fn, i)
        logger.profile_dict[i + 1] = SavedProfile(0, 0, i + 1)

    for i, key in enumerate(jax.random.split(key, n_total_steps // n_rollout_steps)):
        epoch_key, init_key = jax.random.split(key)
        env_state, obs, log, foodlog, phys_state, pponet = epoch(
            env_state,
            obs,
            env,
            pponet,
            reward_fn,
            hazard_fn,
            birth_fn,
            epoch_key,
            n_rollout_steps,
        )

        if visualizer is not None:
            visualizer.render(env_state.physics)  # type: ignore
            visualizer.show()
            popl = jnp.sum(env_state.unique_id.is_active())
            print(f"Population: {int(popl)}")

        # Extinct?
        n_active = jnp.sum(env_state.unique_id.is_active())  # type: ignore
        if n_active == 0:
            print(f"Extinct after {i + 1} epochs")
            break

        # Save network
        log_with_step = log.with_step(i * n_rollout_steps)
        log_death = log_with_step.filter_death()
        logger.save_agents(pponet, log_death.dead, log_death.slots)
        log_birth = log_with_step.filter_birth()
        # Initialize network and adam state for new agents
        is_new = jnp.zeros(env.n_max_agents, dtype=bool).at[log_birth.slots].set(True)
        if jnp.any(is_new):
            pponet = replace_net(init_key, is_new, pponet)

        # Mutation
        reward_fn = rfn.mutate_reward_fn(
            key,
            logger.reward_fn_dict,
            reward_fn,
            mutation,
            log_birth.parents,
            log_birth.unique_id,
            log_birth.slots,
        )
        # Update profile
        for step, uid, parent in zip(
            log_birth.step,
            log_birth.unique_id,
            log_birth.parents,
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
    )


def main(
    seed: int = 1,
    action_cost: float = 4e-5,
    n_rollout_steps: int = 1024,
    n_total_steps: int = 1024 * 10000,
    act_reward_coef: float = 0.001,
    cfconfig_path: Path = PROJECT_ROOT / "config/env/20240224-ls-square.toml",
    bdconfig_path: Path = PROJECT_ROOT / "config/bd/20240318-mild-slope.toml",
    gopsconfig_path: Path = PROJECT_ROOT / "config/gops/20240318-cauchy.toml",
    env_override: str = "",
    birth_override: str = "",
    hazard_override: str = "",
    logdir: Path = Path("./log"),
    log_mode: LogMode = LogMode.REWARD_LOG_STATE,
    log_interval: int = 1000,
    savestate_interval: int = 1000,
    debug_vis: bool = False,
    force_gpu: bool = False,
) -> None:
    if force_gpu and not is_cuda_ready():
        raise RuntimeError("Detected some problem in CUDA!")

    # Load config
    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())
    with bdconfig_path.open("r") as f:
        bdconfig = toml.from_toml(BDConfig, f.read())
    with gopsconfig_path.open("r") as f:
        gopsconfig = toml.from_toml(GopsConfig, f.read())

    # Apply overrides
    cfconfig.apply_override(env_override)
    bdconfig.apply_birth_override(birth_override)
    bdconfig.apply_hazard_override(hazard_override)

    # Load models
    birth_fn, hazard_fn = bdconfig.load_models()
    mutation = gopsconfig.load_model()
    # Override config
    cfconfig.force_energy_consumption = action_cost
    # Make env
    env = make("CircleForaging-v0", **dataclasses.asdict(cfconfig))
    key, reward_key = jax.random.split(jax.random.PRNGKey(seed))
    reward_extracor = RewardExtractor(
        act_space=env.act_space,  # type: ignore
        act_coef=act_reward_coef,
    )
    reward_fn_instance = rfn.LinearReward(
        key=reward_key,
        n_agents=cfconfig.n_max_agents,
        n_weights=1 + cfconfig.n_food_sources,
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
    )
    run_noevo(
        key=key,
        env=env,
        n_initial_agents=cfconfig.n_initial_agents,
        n_rollout_steps=n_rollout_steps,
        n_total_steps=n_total_steps,
        reward_fn=reward_fn_instance,
        hazard_fn=hazard_fn,
        birth_fn=birth_fn,
        mutation=cast(gops.Mutation, mutation),
        xmax=cfconfig.xlim[1],
        ymax=cfconfig.ylim[1],
        logger=logger,
        debug_vis=debug_vis,
    )


if __name__ == "__main__":
    typer.run(main)
