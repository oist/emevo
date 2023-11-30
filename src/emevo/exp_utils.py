"""Utility for experiments"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    Type,
    TypeVar,
)

import serde

def _load_cls(cls_path: str) -> Type:
    try:
        mod, cls = cls_path.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    except (AttributeError, ModuleNotFoundError, ValueError) as err:
        raise ImportError(f"{cls_path} is not a valid class path") from err


@serde
@dataclasses.dataclass(frozen=True)
class BDConfig:
    birth_fn: str
    birth_params: Dict[str, float]
    hazard_fn: str
    hazard_params: Dict[str, float]

    def load_models(self) -> tuple[bd.birth.BirthFunction, bd.death.HazardFunction]:
        birth_fn = _load_cls(self.birth_fn)(**self.birth_params)
        hazard_fn = _load_cls(self.hazard_fn)(**self.hazard_params)
        return birth_fn, hazard_fn
