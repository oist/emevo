"""Utility for experiments"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import functools
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import EllipsisType
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import serde
from numpy.typing import NDArray
from phyjax2d import Position, StateDict

from emevo import birth_and_death as bd
from emevo import genetic_ops as gops
from emevo.environments.circle_foraging import SensorRange
from emevo.eqx_utils import get_slice
from emevo.reward_fn import RewardFn

Self = Any


@serde.serde
@dataclasses.dataclass
class CfConfig:
    n_initial_agents: int = 6
    n_max_agents: int = 100
    n_max_foods: int = 40
    n_food_sources: int = 1
    food_num_fn: str | tuple[Any, ...] = "constant"
    food_loc_fn: str | tuple[Any, ...] = "gaussian"
    agent_loc_fn: str | tuple[Any, ...] = "uniform"
    food_energy_coef: tuple[float | tuple[float, ...], ...] = (1.0,)
    food_color: tuple[tuple[int, int, int, int], ...] = ((254, 2, 162, 255),)
    xlim: tuple[float, float] = (0.0, 200.0)
    ylim: tuple[float, float] = (0.0, 200.0)
    env_radius: float = 120.0
    env_shape: str = "square"
    obstacles: str = "none"
    newborn_loc: str = "neighbor"
    neighbor_stddev: float = 40.0
    n_agent_sensors: int = 16
    sensor_length: float = 100.0
    sensor_range: SensorRange = SensorRange.WIDE
    agent_radius: float = 10.0
    food_radius: float = 4.0
    foodloc_interval: int = 1000
    fec_intervals: tuple[int, ...] = (1,)
    dt: float = 0.1
    linear_damping: float = 0.8
    angular_damping: float = 0.6
    max_force: float = 40.0
    min_force: float = -20.0
    init_energy: float = 20.0
    energy_capacity: float = 100.0
    force_energy_consumption: float = 0.01 / 40.0
    basic_energy_consumption: float = 0.0
    energy_share_ratio: float = 0.4
    n_velocity_iter: int = 6
    n_position_iter: int = 2
    n_physics_iter: int = 5
    max_place_attempts: int = 10
    observe_food_label: bool = False
    mouth_range: str | list[int] = "front"
    n_tactile_bins: int = 6
    tactile_shift: float = 0.0
    foods_min_dist: float = 0.0

    def apply_override(self, override: str) -> None:
        if 0 < len(override):
            override_dict = json.loads(override)
            for key, value in override_dict.items():
                setattr(self, key, value)


def _load_cls(cls_path: str) -> type:
    try:
        mod, cls = cls_path.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    except (AttributeError, ModuleNotFoundError, ValueError) as err:
        raise ImportError(f"{cls_path} is not a valid class path") from err


@serde.serde(type_check=serde.disabled)
@dataclasses.dataclass
class BDConfig:
    birth_fn: str
    birth_params: dict[str, float]
    hazard_fn: str
    hazard_params: dict[str, float]

    def load_models(self) -> tuple[bd.BirthFunction, bd.HazardFunction]:
        birth_fn = _load_cls(self.birth_fn)(**self.birth_params)
        hazard_fn = _load_cls(self.hazard_fn)(**self.hazard_params)
        return birth_fn, hazard_fn

    def apply_birth_override(self, override: str) -> None:
        if 0 < len(override):
            override_dict = json.loads(override)
            self.birth_params |= override_dict

    def apply_hazard_override(self, override: str) -> None:
        if 0 < len(override):
            override_dict = json.loads(override)
            self.hazard_params |= override_dict


def _resolve_cls(d: dict[str, Any]) -> GopsConfig:
    params = {}
    for k, v in d["params"].items():
        if isinstance(v, dict):
            params[k] = _resolve_cls(v)
        else:
            params[k] = v
    return _load_cls(d["path"])(**d["params"])


@serde.serde
@dataclasses.dataclass(frozen=True)
class GopsConfig:
    path: str
    init_std: float
    init_mean: float
    params: dict[str, float | dict[str, Any]]
    init_kwargs: dict[str, float] = dataclasses.field(default_factory=dict)

    def load_model(self) -> gops.Mutation | gops.Crossover:
        params = {}
        for k, v in self.params.items():
            if isinstance(v, dict):
                params[k] = _resolve_cls(v)
            else:
                params[k] = v
        return _load_cls(self.path)(**params)

    def apply_params_override(self, override: str) -> None:
        if 0 < len(override):
            override_dict = json.loads(override)
            for key, value in override_dict.items():
                self.params[key] = value


@chex.dataclass
class Log:
    dead: jax.Array
    n_got_food: jax.Array
    action_magnitude: jax.Array
    energy_gain: jax.Array
    consumed_energy: jax.Array
    energy: jax.Array
    parents: jax.Array
    rewards: jax.Array
    unique_id: jax.Array
    additional_fields: dict[str, jax.Array] = dataclasses.field(default_factory=dict)

    def with_step(self, from_: int) -> LogWithStep:
        if self.parents.ndim == 2:
            step_size, batch_size = self.parents.shape
            step_arange = jnp.arange(from_, from_ + step_size)
            step = jnp.tile(jnp.expand_dims(step_arange, axis=1), (1, batch_size))
            slots_arange = jnp.arange(batch_size)
            slots = jnp.tile(slots_arange, (step_size, 1))
            return LogWithStep(log=self, step=step, slots=slots)
        elif self.parents.ndim == 1:
            batch_size = self.parents.shape[0]
            return LogWithStep(
                log=self,
                step=jnp.ones(batch_size, dtype=jnp.int32) * from_,
                slots=jnp.arange(batch_size),
            )
        else:
            raise ValueError(
                "with_step is only applicable for 1 or 2 dimensional log, but it has"
                + f"{self.parents.ndim} ndim"
            )


@chex.dataclass
class FoodLog:
    eaten: jax.Array  # i32, [N_FOOD_SOURCES,]
    regenerated: jax.Array  # i32, [N_FOOD_SOURCES,]


@chex.dataclass
class LogWithStep:
    log: Log
    step: jax.Array
    slots: jax.Array

    def filter_active(self) -> Any:
        is_active = self.log.unique_id > 0
        return jax.tree_util.tree_map(lambda arr: arr[is_active], self)

    def filter_birth(self) -> Any:
        is_birth_event = self.log.parents > 0
        return jax.tree_util.tree_map(lambda arr: arr[is_birth_event], self)

    def filter_death(self) -> Any:
        is_death_event = self.log.dead > 0
        return jax.tree_util.tree_map(lambda arr: arr[is_death_event], self)

    def to_flat_dict(self) -> dict[str, jax.Array]:
        d = dataclasses.asdict(self.log)  # type: ignore
        additional = d.pop("additional_fields")
        d["step"] = self.step
        d["slots"] = self.slots
        return d | additional


@dataclasses.dataclass
class SavedProfile:
    birthtime: int
    parent: int
    unique_id: int


_XY_SAVE_DTYPE = np.float16


@chex.dataclass
class SavedPhysicsState:
    circle_axy: jax.Array
    circle_is_active: jax.Array
    static_circle_axy: jax.Array
    static_circle_is_active: jax.Array
    static_circle_label: jax.Array

    @staticmethod
    def load(path: Path) -> Self:
        npzfile = np.load(path)
        static_circle_is_active = jnp.array(npzfile["static_circle_is_active"])
        # For backward compatibility
        if "static_circle_label" in npzfile:
            static_circle_label = jnp.array(npzfile["static_circle_label"])
        else:
            static_circle_label = jnp.zeros(
                static_circle_is_active.shape[0],
                dtype=jnp.uint8,
            )
        return SavedPhysicsState(
            circle_axy=jnp.array(npzfile["circle_axy"].astype(np.float32)),
            circle_is_active=jnp.array(npzfile["circle_is_active"]),
            static_circle_axy=jnp.array(
                npzfile["static_circle_axy"].astype(np.float32)
            ),
            static_circle_is_active=static_circle_is_active,
            static_circle_label=static_circle_label,
        )

    def set_by_index(
        self,
        i: int | EllipsisType | tuple[int, int],
        phys: StateDict,
    ) -> StateDict:
        phys = phys.nested_replace(
            "circle.p",
            Position.from_axy(self.circle_axy[i]),
        )
        phys = phys.nested_replace("circle.is_active", self.circle_is_active[i])
        phys = phys.nested_replace(
            "static_circle.p",
            Position.from_axy(self.static_circle_axy[i]),
        )
        phys = phys.nested_replace(
            "static_circle.is_active",
            self.static_circle_is_active[i],
        )
        phys = phys.nested_replace(
            "static_circle.label",
            self.static_circle_label[i],
        )
        return phys


def save_physstates(phys_states: list[SavedPhysicsState], path: Path) -> None:
    concatenated = jax.tree_util.tree_map(
        lambda *args: np.concatenate(args),
        *phys_states,
    )
    np.savez_compressed(
        path,
        circle_axy=concatenated.circle_axy.astype(_XY_SAVE_DTYPE),
        circle_is_active=concatenated.circle_is_active,
        static_circle_axy=concatenated.static_circle_axy.astype(_XY_SAVE_DTYPE),
        static_circle_is_active=concatenated.static_circle_is_active,
        static_circle_label=concatenated.static_circle_label,
    )


class LogMode(str, enum.Enum):
    NONE = "none"
    REWARD = "reward"
    REWARD_LOG = "reward-log"
    REWARD_LOG_STATE = "reward-log-state"
    FULL = "reward-log-state-agent"


def _default_dropped_keys() -> list[str]:
    return ["dead", "parents"]


@dataclasses.dataclass
class Logger:
    logdir: Path
    mode: LogMode
    log_interval: int
    savestate_interval: int
    min_age_for_save: int
    dropped_keys: list[str] = dataclasses.field(default_factory=_default_dropped_keys)
    reward_fn_dict: dict[int, RewardFn] = dataclasses.field(default_factory=dict)
    profile_dict: dict[int, SavedProfile] = dataclasses.field(default_factory=dict)
    _log_list: list[dict[str, NDArray]] = dataclasses.field(
        default_factory=list,
        init=False,
    )
    _foodlog_list: list[FoodLog] = dataclasses.field(default_factory=list, init=False)
    _physstate_list: list[SavedPhysicsState] = dataclasses.field(
        default_factory=list,
        init=False,
    )
    _log_index: int = dataclasses.field(default=1, init=False)
    _physstate_index: int = dataclasses.field(default=1, init=False)

    def push_log(self, log: LogWithStep) -> None:
        if "log" not in self.mode.value:
            return

        # Move log to CPU
        logd = log.to_flat_dict()
        self._log_list.append(jax.tree_util.tree_map(np.array, logd))
        if len(self._log_list) % self.log_interval == 0:
            self._save_log()

    def _save_log(self) -> None:
        if len(self._log_list) == 0:
            return

        all_log = jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=0),
            *self._log_list,
        )
        log_dict = {}

        for key, value in all_log.items():
            if key in self.dropped_keys:
                continue
            if value.ndim == 2 and value.shape[1] > 1:
                for i in range(value.shape[1]):
                    log_dict[f"{key}_{i + 1}"] = value[:, i]
            elif value.ndim >= 2:
                log_dict[key] = value.ravel()
            else:
                log_dict[key] = value

        pq.write_table(
            pa.Table.from_pydict(log_dict),
            self.logdir.joinpath(f"log-{self._log_index}.parquet"),
            compression="zstd",
        )
        self._log_index += 1
        self._log_list.clear()

    def push_foodlog(self, log: FoodLog) -> None:
        if "log" not in self.mode.value:
            return

        # Move log to CPU
        self._foodlog_list.append(jax.tree_util.tree_map(np.array, log))

        if len(self._foodlog_list) % self.log_interval == 0:
            self._save_foodlog()

    def _save_foodlog(self) -> None:
        if len(self._foodlog_list) == 0:
            return

        all_log = jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=0),
            *self._foodlog_list,
        )
        log_dict = {}
        for i in range(all_log.eaten.shape[1]):
            log_dict[f"eaten_{i}"] = all_log.eaten[:, i]
            log_dict[f"regen_{i}"] = all_log.regenerated[:, i]

        # Don't change log_index here
        pq.write_table(
            pa.Table.from_pydict(log_dict),
            self.logdir.joinpath(f"foodlog-{self._log_index}.parquet"),
            compression="zstd",
        )
        self._foodlog_list.clear()

    def push_physstate(self, phys_state: SavedPhysicsState) -> None:
        if "state" not in self.mode.value:
            return

        # Move it to CPU to save memory
        self._physstate_list.append(jax.tree_util.tree_map(np.array, phys_state))

        if len(self._physstate_list) % self.savestate_interval == 0:
            self._save_physstate()

    def _save_physstate(self) -> None:
        if len(self._physstate_list) == 0:
            return

        save_physstates(
            self._physstate_list,
            self.logdir.joinpath(f"state-{self._physstate_index}.npz"),
        )
        self._physstate_index += 1
        self._physstate_list.clear()

    def save_agents(
        self,
        net: eqx.Module,
        unique_id: jax.Array,
        slots: jax.Array,
        ages: jax.Array,
        prefix: str = "dead",
    ) -> None:
        if "agent" not in self.mode.value:
            return

        for uid, slot, age in zip(np.array(unique_id), np.array(slots), np.array(ages)):
            if age < self.min_age_for_save:
                continue
            sliced_net = get_slice(net, slot)
            modelpath = self.logdir.joinpath(f"{prefix}-{uid}-age{age}.eqx")
            eqx.tree_serialise_leaves(modelpath, sliced_net)

    def save_profile_and_rewards(self) -> None:
        profile_and_rewards = [
            v.serialise() | dataclasses.asdict(self.profile_dict[k])
            for k, v in self.reward_fn_dict.items()
        ]
        table = pa.Table.from_pylist(profile_and_rewards)
        pq.write_table(table, self.logdir.joinpath("profile_and_rewards.parquet"))

    def finalize(self) -> None:
        if "reward" in self.mode.value:
            self.save_profile_and_rewards()

        if "log" in self.mode.value:
            self._save_foodlog()
            self._save_log()

        if "state" in self.mode.value:
            self._save_physstate()


def simple_profiler(fn: Callable[..., Any]) -> Any:
    """Quite simple profiling decorator. It's somewhat useful for debugging."""

    @functools.wraps(fn)
    def wrapper(*args, **kargs) -> Any:
        before = dt.datetime.now()
        ret = fn(*args, **kargs)
        after = dt.datetime.now()
        elapsed = after - before
        print(elapsed)
        return ret

    return wrapper


def is_cuda_ready() -> bool:
    try:
        _ = jax.device_put(jnp.zeros(1), device=jax.devices("gpu")[0])
        return True
    except Exception as e:
        print(e, file=sys.stderr)
        return False
