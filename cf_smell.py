""" Asexual reward evolution with CircleForagingWithSmell"""
import dataclasses
import enum
import json
from pathlib import Path
from typing import Optional, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from serde import toml

from emevo import Env
from emevo import birth_and_death as bd
from emevo import genetic_ops as gops
from emevo import make
from emevo.env import ObsProtocol as Obs
from emevo.env import StateProtocol as State
from emevo.eqx_utils import get_slice
from emevo.eqx_utils import where as eqx_where
from emevo.exp_utils import (
    BDConfig,
    CfConfig,
    GopsConfig,
    Log,
    Logger,
    LogMode,
    SavedPhysicsState,
    SavedProfile,
)
from emevo.reward_fn import (
    ExponentialReward,
    LinearReward,
    RewardFn,
    SigmoidReward,
    SigmoidReward_01,
    SinhReward,
    mutate_reward_fn,
    serialize_weight,
)
from emevo.rl.ppo_normal import (
    NormalPPONet,
    Rollout,
    vmap_apply,
    vmap_batch,
    vmap_net,
    vmap_update,
    vmap_value,
)
from emevo.spaces import BoxSpace
from emevo.visualizer import SaveVideoWrapper

PROJECT_ROOT = Path(__file__).parent.parent


class RewardKind(str, enum.Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"
    SIGMOID_01 = "sigmoid-01"
    SINH = "sinh"


@dataclasses.dataclass
class RewardExtractor:
    act_space: BoxSpace
    act_coef: float
    smell_coef: float
    mask: dataclasses.InitVar[str] = "11111"
    _mask_array: jax.Array = dataclasses.field(init=False)
    _max_norm: jax.Array = dataclasses.field(init=False)

    def __post_init__(self, mask: str) -> None:
        mask_array = jnp.array([x == "1" for x in mask])
        self._mask_array = jnp.expand_dims(mask_array, axis=0)
        self._max_norm = jnp.sqrt(
            jnp.sum(self.act_space.high**2, axis=-1, keepdims=True)
        )

    def normalize_action(self, action: jax.Array) -> jax.Array:
        scaled = self.act_space.sigmoid_scale(action)
        norm = jnp.sqrt(jnp.sum(scaled**2, axis=-1, keepdims=True))
        return norm / self._max_norm

    def extract_linear(
        self,
        collision: jax.Array,
        action: jax.Array,
        energy: jax.Array,
        smell: jax.Array,
    ) -> jax.Array:
        del energy
        act_input = self.act_coef * self.normalize_action(action)
        smell_input = self.smell_coef * smell
        return (
            jnp.concatenate((collision, act_input, smell_input), axis=1)
            * self._mask_array
        )

    def extract_sigmoid(
        self,
        collision: jax.Array,
        action: jax.Array,
        energy: jax.Array,
        smell: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        act_input = self.act_coef * self.normalize_action(action)
        smell_input = self.smell_coef * smell
        reward_input = jnp.concatenate((collision, act_input, smell_input), axis=1)
        return reward_input * self._mask_array, energy


def linear_rs(w: jax.Array) -> dict[str, jax.Array]:
    return serialize_weight(w, ["agent", "food", "poison", "wall", "action", "smell"])


def exp_rs(w: jax.Array, scale: jax.Array) -> dict[str, jax.Array]:
    w_dict = serialize_weight(w, ["w_agent", "w_food", "w_wall", "w_action", "w_smell"])
    scale_dict = serialize_weight(
        scale,
        ["scale_agent", "scale_food", "scale_wall", "scale_action", "scale_smell"],
    )
    return w_dict | scale_dict


def sigmoid_rs(w: jax.Array, alpha: jax.Array) -> dict[str, jax.Array]:
    w_dict = serialize_weight(w, ["w_agent", "w_food", "w_wall", "w_action", "w_smell"])
    alpha_dict = serialize_weight(
        alpha,
        ["alpha_agent", "alpha_food", "alpha_wall", "alpha_action", "alpha_smell"],
    )
    return w_dict | alpha_dict


def linear_rs_withp(w: jax.Array) -> dict[str, jax.Array]:
    return serialize_weight(
        w,
        ["agent", "food", "poison", "wall", "action", "fsmell", "psmell"],
    )


def exp_rs_withp(w: jax.Array, scale: jax.Array) -> dict[str, jax.Array]:
    w_dict = serialize_weight(
        w,
        ["w_agent", "w_food", "w_poison", "w_wall", "w_action", "w_fsmell", "w_psmell"],
    )
    scale_dict = serialize_weight(
        scale,
        [
            "scale_agent",
            "scale_food",
            "scale_poison",
            "scale_wall",
            "scale_action",
            "scale_fsmell",
            "scale_psmell",
        ],
    )
    return w_dict | scale_dict


def sigmoid_rs_withp(w: jax.Array, alpha: jax.Array) -> dict[str, jax.Array]:
    w_dict = serialize_weight(
        w,
        ["w_agent", "w_food", "w_poison", "w_wall", "w_action", "w_fsmell", "w_psmell"],
    )
    alpha_dict = serialize_weight(
        alpha,
        [
            "alpha_agent",
            "alpha_food",
            "alpha_poison",
            "alpha_wall",
            "alpha_action",
            "alpha_fsmell",
            "alpha_psmell",
        ],
    )
    return w_dict | alpha_dict


def exec_rollout(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    reward_fn: RewardFn,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    prng_key: jax.Array,
    n_rollout_steps: int,
) -> tuple[State, Rollout, Log, SavedPhysicsState, Obs, jax.Array]:
    def step_rollout(
        carried: tuple[State, Obs],
        key: jax.Array,
    ) -> tuple[tuple[State, Obs], tuple[Rollout, Log, SavedPhysicsState]]:
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
        smell = state_t.smell  # type: ignore
        rewards = reward_fn(obs_t1.collision, actions, energy, smell).reshape(-1, 1)
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
            jnp.logical_and(
                jnp.logical_not(dead),
                state.unique_id.is_active(),  # type: ignore
            ),
            jax.random.bernoulli(birth_key, p=birth_prob),
        )
        state_t1db, parents = env.activate(state_t1d, possible_parents)
        log = Log(
            dead=jnp.where(dead, state_t.unique_id.unique_id, -1),  # type: ignore
            got_food=obs_t1.collision[:, 1],
            parents=parents,
            rewards=rewards.ravel(),
            age=state_t1db.status.age,
            energy=state_t1db.status.energy,
            unique_id=state_t1db.unique_id.unique_id,
        )
        phys = state_t.physics  # type: ignore
        phys_state = SavedPhysicsState(
            circle_axy=phys.circle.p.into_axy(),
            static_circle_axy=phys.static_circle.p.into_axy(),
            circle_is_active=phys.circle.is_active,
            static_circle_is_active=phys.static_circle.is_active,
            static_circle_label=phys.static_circle.label,
        )
        return (state_t1db, obs_t1), (rollout, log, phys_state)

    (state, obs), (rollout, log, phys_state) = jax.lax.scan(
        step_rollout,
        (state, initial_obs),
        jax.random.split(prng_key, n_rollout_steps),
    )
    next_value = vmap_value(network, obs.as_array())
    return state, rollout, log, phys_state, obs, next_value


@eqx.filter_jit
def epoch(
    state: State,
    initial_obs: Obs,
    env: Env,
    network: NormalPPONet,
    reward_fn: RewardFn,
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
) -> tuple[State, Obs, Log, SavedPhysicsState, optax.OptState, NormalPPONet]:
    keys = jax.random.split(prng_key, env.n_max_agents + 1)
    env_state, rollout, log, phys_state, obs, next_value = exec_rollout(
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
    return env_state, obs, log, phys_state, opt_state, pponet


def run_evolution(
    *,
    key: jax.Array,
    env: Env,
    n_initial_agents: int,
    adam: optax.GradientTransformation,
    gamma: float,
    gae_lambda: float,
    n_optim_epochs: int,
    minibatch_size: int,
    n_rollout_steps: int,
    n_total_steps: int,
    reward_fn: RewardFn,
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
    adam_init, adam_update = adam

    @eqx.filter_jit
    def initialize_opt_state(net: eqx.Module) -> optax.OptState:
        return jax.vmap(adam_init)(eqx.filter(net, eqx.is_array))

    @eqx.filter_jit
    def replace_net(
        key: chex.PRNGKey,
        flag: jax.Array,
        pponet: NormalPPONet,
        opt_state: optax.OptState,
    ) -> tuple[NormalPPONet, optax.OptState]:
        initialized = initialize_net(key)
        pponet = eqx_where(flag, initialized, pponet)
        opt_state = jax.tree_map(
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
        env_state, obs, log, phys_state, opt_state, pponet = epoch(
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
        )

        if visualizer is not None:
            visualizer.render(env_state.physics)  # type: ignore
            visualizer.show()
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
            pponet, opt_state = replace_net(init_key, is_new, pponet, opt_state)

        # Mutation
        reward_fn = mutate_reward_fn(
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


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def evolve(
    seed: int = 1,
    n_agents: int = 20,
    init_energy: float = 20.0,
    action_cost: float = 0.0001,
    mutation_prob: float = 0.2,
    adam_lr: float = 3e-4,
    adam_eps: float = 1e-7,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    n_optim_epochs: int = 10,
    minibatch_size: int = 256,
    n_rollout_steps: int = 1024,
    n_total_steps: int = 1024 * 10000,
    act_reward_coef: float = 0.001,
    cfconfig_path: Path = PROJECT_ROOT / "config/env/20231214-square.toml",
    bdconfig_path: Path = PROJECT_ROOT / "config/bd/20230530-a035-e020.toml",
    gopsconfig_path: Path = PROJECT_ROOT / "config/gops/20240111-mutation-0401.toml",
    env_override: str = "",
    birth_override: str = "",
    hazard_override: str = "",
    reward_mask: Optional[str] = None,
    reward_fn: RewardKind = RewardKind.LINEAR,
    logdir: Path = Path("./log"),
    log_mode: LogMode = LogMode.FULL,
    log_interval: int = 1000,
    savestate_interval: int = 1000,
    poison_reward: bool = False,
    debug_vis: bool = False,
) -> None:
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

    if reward_mask is None:
        if poison_reward:
            reward_mask = "1111111"
        else:
            reward_mask = "11111"

    # Load models
    birth_fn, hazard_fn = bdconfig.load_models()
    mutation = gopsconfig.load_model()
    # Override config
    cfconfig.n_initial_agents = n_agents
    cfconfig.init_energy = init_energy
    cfconfig.force_energy_consumption = action_cost
    gopsconfig.params["mutation_prob"] = mutation_prob
    # Make env
    env = make("CircleForaging-v1", **dataclasses.asdict(cfconfig))
    key, reward_key = jax.random.split(jax.random.PRNGKey(seed))
    reward_extracor = RewardExtractor(
        act_space=env.act_space,  # type: ignore
        act_coef=act_reward_coef,
        smell_coef=1.0 / cfconfig.n_max_foods,
        mask=reward_mask,
    )
    common_rewardfn_args = {
        "key": reward_key,
        "n_agents": cfconfig.n_max_agents,
        "n_weights": 7 if poison_reward else 5,
        "std": gopsconfig.init_std,
        "mean": gopsconfig.init_mean,
    }
    if reward_fn == RewardKind.LINEAR:
        reward_fn_instance = LinearReward(
            **common_rewardfn_args,
            extractor=reward_extracor.extract_linear,
            serializer=linear_rs_withp if poison_reward else linear_rs,
        )
    elif reward_fn == RewardKind.EXPONENTIAL:
        reward_fn_instance = ExponentialReward(
            **common_rewardfn_args,
            extractor=reward_extracor.extract_linear,
            serializer=exp_rs_withp if poison_reward else exp_rs,
        )
    elif reward_fn == RewardKind.SIGMOID:
        reward_fn_instance = SigmoidReward(
            **common_rewardfn_args,
            extractor=reward_extracor.extract_sigmoid,
            serializer=sigmoid_rs_withp if poison_reward else sigmoid_rs,
        )
    elif reward_fn == RewardKind.SIGMOID_01:
        reward_fn_instance = SigmoidReward_01(
            **common_rewardfn_args,
            extractor=reward_extracor.extract_sigmoid,
            serializer=sigmoid_rs_withp if poison_reward else sigmoid_rs,
        )
    elif reward_fn == RewardKind.SINH:
        reward_fn_instance = SinhReward(
            **common_rewardfn_args,
            extractor=reward_extracor.extract_linear,
            serializer=linear_rs_withp if poison_reward else linear_rs,
        )
    else:
        raise ValueError(f"Invalid reward_fn {reward_fn}")

    logger = Logger(
        logdir=logdir,
        mode=log_mode,
        log_interval=log_interval,
        savestate_interval=savestate_interval,
    )
    run_evolution(
        key=key,
        env=env,
        n_initial_agents=n_agents,
        adam=optax.adam(adam_lr, eps=adam_eps),
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_optim_epochs=n_optim_epochs,
        minibatch_size=minibatch_size,
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


@app.command()
def replay(
    physstate_path: Path,
    backend: str = "pyglet",  # Use "headless" for headless rendering
    videopath: Optional[Path] = None,
    start: int = 0,
    end: Optional[int] = None,
    cfconfig_path: Path = PROJECT_ROOT / "config/env/20231214-square.toml",
    env_override: str = "",
) -> None:
    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())
    # For speedup
    cfconfig.n_initial_agents = 1
    cfconfig.apply_override(env_override)
    phys_state = SavedPhysicsState.load(physstate_path)
    env = make("CircleForaging-v1", **dataclasses.asdict(cfconfig))
    env_state, _ = env.reset(jax.random.PRNGKey(0))
    end_index = end if end is not None else phys_state.circle_axy.shape[0]
    visualizer = env.visualizer(
        env_state,
        figsize=(cfconfig.xlim[1] * 2, cfconfig.ylim[1] * 2),
        backend=backend,
    )
    if videopath is not None:
        visualizer = SaveVideoWrapper(visualizer, videopath, fps=60)
    for i in range(start, end_index):
        phys = phys_state.set_by_index(i, env_state.physics)
        env_state = dataclasses.replace(env_state, physics=phys)
        visualizer.render(env_state.physics)
        visualizer.show()
    visualizer.close()


@app.command()
def widget(
    physstate_path: Path,
    start: int = 0,
    end: Optional[int] = None,
    cfconfig_path: Path = PROJECT_ROOT / "config/env/20231214-square.toml",
    log_offset: int = 0,
    log_path: Optional[Path] = None,
    self_terminate: bool = False,
    profile_and_rewards_path: Optional[Path] = None,
    cm_fixed_minmax: str = "",
    env_override: str = "",
) -> None:
    from emevo.analysis.qt_widget import CFEnvReplayWidget, start_widget

    with cfconfig_path.open("r") as f:
        cfconfig = toml.from_toml(CfConfig, f.read())
    # For speedup
    cfconfig.n_initial_agents = 1
    cfconfig.apply_override(env_override)
    phys_state = SavedPhysicsState.load(physstate_path)
    env = make("CircleForaging-v1", **dataclasses.asdict(cfconfig))
    end = phys_state.circle_axy.shape[0] if end is None else end
    if log_path is None:
        log_ds = None
        step_offset = 0
    else:
        import pyarrow.dataset as ds

        log_ds = ds.dataset(log_path)
        first_step = log_ds.scanner(columns=["step"]).head(1)["step"][0].as_py()
        step_offset = first_step + log_offset

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
    )


if __name__ == "__main__":
    app()