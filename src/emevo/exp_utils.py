"""Utility for experiments"""
from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Dict, Tuple, Type, Union

import chex
import jax
import jax.numpy as jnp
import serde

from emevo import birth_and_death as bd
from emevo import genetic_ops as gops
from emevo.environments.circle_foraging import SensorRange


@serde.serde
@dataclasses.dataclass
class CfConfig:
    n_initial_agents: int = 6
    n_max_agents: int = 100
    n_max_foods: int = 40
    food_num_fn: Union[str, Tuple[str, ...]] = "constant"
    food_loc_fn: Union[str, Tuple[str, ...]] = "gaussian"
    agent_loc_fn: Union[str, Tuple[str, ...]] = "uniform"
    xlim: Tuple[float, float] = (0.0, 200.0)
    ylim: Tuple[float, float] = (0.0, 200.0)
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
    dt: float = 0.1
    linear_damping: float = 0.8
    angular_damping: float = 0.6
    max_force: float = 40.0
    min_force: float = -20.0
    init_energy: float = 20.0
    energy_capacity: float = 100.0
    force_energy_consumption: float = 0.01 / 40.0
    energy_share_ratio: float = 0.4
    n_velocity_iter: int = 6
    n_position_iter: int = 2
    n_physics_iter: int = 5
    max_place_attempts: int = 10


def _load_cls(cls_path: str) -> Type:
    try:
        mod, cls = cls_path.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    except (AttributeError, ModuleNotFoundError, ValueError) as err:
        raise ImportError(f"{cls_path} is not a valid class path") from err


@serde.serde
@dataclasses.dataclass(frozen=True)
class BDConfig:
    birth_fn: str
    birth_params: Dict[str, float]
    hazard_fn: str
    hazard_params: Dict[str, float]

    def load_models(self) -> tuple[bd.BirthFunction, bd.HazardFunction]:
        birth_fn = _load_cls(self.birth_fn)(**self.birth_params)
        hazard_fn = _load_cls(self.hazard_fn)(**self.hazard_params)
        return birth_fn, hazard_fn


def _resolve_cls(d: dict[str, Any]) -> GopsConfig:
    params = {}
    for k, v in d["params"].items():
        if isinstance(v, dict):
            params[k] = _resolve_cls(v)
        else:
            params[k] = v
    return _load_cls(d["path"], **d["params"])


@serde.serde
@dataclasses.dataclass(frozen=True)
class GopsConfig:
    path: str
    params: Dict[str, Union[float, Dict[str, float]]]

    def load_model(self) -> gops.Mutation | gops.Crossover:
        params = {}
        for k, v in params.items():
            if isinstance(v, dict):
                params[k] = _resolve_cls(v)
            else:
                params[k] = v
        return _load_cls(self.path)(**params)


@chex.dataclass
class Log:
    parents: jax.Array
    rewards: jax.Array
    age: jax.Array
    energy: jax.Array
    birthtime: jax.Array
    generation: jax.Array
    unique_id: jax.Array

    def with_step(self, from_: int) -> LogWithStep:
        if self.parents.ndim == 2:
            step_size, batch_size = self.parents.shape
            arange = jnp.arange(from_, from_ + step_size)
            step = jnp.tile(jnp.expand_dims(arange, axis=1), (1, batch_size))
            return LogWithStep(**dataclasses.asdict(self), step=step)
        elif self.parents.ndim == 1:
            batch_size = self.parents.shape[0]
            return LogWithStep(
                **dataclasses.asdict(self),
                step=jnp.ones(batch_size, dtype=jnp.int32) * from_,
            )
        else:
            raise ValueError(
                "with_step is only applicable for 1 or 2 dimensional log, but it has"
                + f"{self.parents.ndim} ndim"
            )


@chex.dataclass
class LogWithStep(Log):
    step: jax.Array

    def filter(self) -> Any:
        is_active = self.unique_id > -1
        return jax.tree_map(lambda arr: arr[is_active], self)
