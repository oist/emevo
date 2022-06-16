""" Gym-like make/register system
"""
import dataclasses
from typing import Any, Dict, NoReturn, Optional, Type

import numpy as np

from emevo import Env


@dataclasses.dataclass(frozen=True)
class _EnvSpec:
    cls: Type[Env]
    description: Optional[str]
    default_kwargs: Dict[str, Any]


_REGISTERED_ENVS: Dict[str, _EnvSpec] = {}


def _levenshtein_distance(s_: str, t_: str) -> int:
    s = np.array(bytearray(s_, encoding="utf-8"))
    t = np.array(bytearray(t_, encoding="utf-8"))
    n = len(t)
    v0 = np.arange(n + 1, dtype=np.int32)
    v1 = np.zeros(n + 1, dtype=np.int32)

    for i, c in enumerate(s):
        v1[0] = i + 1
        for j in range(n):
            del_cost = v0[j + 1] + 1
            ins_cost = v1[j] + 1
            sub_cost = v0[j] if c == t[j] else v0[j] + 1
            v1[j + 1] = min(del_cost, ins_cost, sub_cost)
        v0, v1 = v1, v0
    return v0[-1]


def _raise_noenv_error(name: str) -> NoReturn:
    THRESHOLD = 4
    best = None, int(1e9)
    for env_name in _REGISTERED_ENVS.keys():
        dist = _levenshtein_distance(name, env_name)
        if dist <= THRESHOLD and dist < best[1]:
            best = env_name, dist
    msg = f"Enviroment name {name} is not registered"
    candidate, _ = best
    if candidate is not None:
        msg += f"\n Do you mean '{candidate}'?"
    raise ValueError(msg)


def description(env_name: str) -> str:
    env_spec = _REGISTERED_ENVS.get(env_name, None)
    if env_spec is None:
        _raise_noenv_error(env_name)
    msg = (
        "Environment specification:\n"
        + f"    name: {env_name}\n "
        + f"    class: {env_spec.cls}"
    )
    if env_spec.description is not None:
        msg += f"\n    description: {env_spec.description} \n"
    return msg


def make(
    env_name: str,
    *args,
    **kwargs,
) -> Env:
    env_spec = _REGISTERED_ENVS.get(env_name, None)
    if env_spec is None:
        _raise_noenv_error(env_name)
    return env_spec.cls(*args, **dict(env_spec.default_kwargs, **kwargs))


def register(
    name: str,
    env_class: Type[Env],
    description: Optional[str] = None,
    default_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    if name in _REGISTERED_ENVS:
        raise ValueError(f"{name} is already registered")
    if default_kwargs is None:
        default_kwargs = {}
    _REGISTERED_ENVS[name] = _EnvSpec(env_class, description, default_kwargs)
