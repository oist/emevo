"""Utility for experiments"""
from __future__ import annotations

import dataclasses
import importlib
from typing import Dict, List, Tuple, Type

import serde

from emevo import birth_and_death as bd


@serde.serde
@dataclasses.dataclass
class CfConfig:
    agent_radius: float
    n_agents: int
    n_agent_sensors: int
    sensor_length: float
    food_loc_fn: str
    food_num_fn: Tuple["str", int]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    env_radius: float
    env_shape: str
    obstacles: List[Tuple[float, float, float, float]]
    seed: int
    linear_damping: float = 0.8
    angular_damping: float = 0.6
    max_force: float = 40.0
    min_force: float = -20.0


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
