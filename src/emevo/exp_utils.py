"""Utility for experiments"""
from __future__ import annotations

import dataclasses
import importlib
from typing import Dict, Tuple, Type, Union

import chex
import jax
import serde

from emevo import birth_and_death as bd
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


@chex.dataclass
class Log:
    parents: jax.Array
    rewards: jax.Array
    dead: jax.Array
    age: jax.Array
    energy: jax.Array
    birthtime: jax.Array
    generation: jax.Array
    unique_id: jax.Array

    @staticmethod
    def avro_schema() -> dict:
        """Apache avro schema for this class"""

        def array(dtype: str) -> dict[str, str]:
            return {"type": "array", "items": dtype}

        return {
            "doc": "Default log schema for emevo",
            "name": "Log",
            "namespace": "emevo",
            "type": "record",
            "fields": [
                {"name": "parents", "type": array("int")},
                {"name": "rewards", "type": array("float")},
                {"name": "dead", "type": array("int")},
                {"name": "energy", "type": array("float")},
                {"name": "age", "type": array("int")},
                {"name": "birthtime", "type": array("int")},
                {"name": "generation", "type": array("int")},
                {"name": "unique_id", "type": array("int")},
            ],
        }
