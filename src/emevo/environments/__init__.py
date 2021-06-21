""" Bultin implementations of emevo environments
"""
import dataclasses
import typing as t

import numpy as np

from emevo import Environment
from .waterworld import WaterWorld


@dataclasses.dataclass(frozen=True)
class _EnviromentSpec:
    cls: t.Type[Environment]
    description: t.Optional[str]


class _Registory:
    """An internal class to register and make environments."""

    registered_envs: t.ClassVar[t.Dict[str, _EnviromentSpec]] = {}


def _levenshtein_distance(s: str, t: str) -> int:
    s = np.array(bytearray(s, encoding="utf-8"))
    t = np.array(bytearray(t, encoding="utf-8"))
    n = len(t)
    v0 = np.arange(n + 1, dtype=np.int32)
    v1 = np.zeros(n + 1, dtype=np.int32)

    for i, c in enumerate(s):
        v1[0] = i + 1
        for j in range(n):
            del_cost = v0[j + 1] + 1
            ins_cost = v1[j] + 1
            sub_cost = v0[j] if s[i] == t[j] else v0[j] + 1
            v1[j + 1] = min(del_cost, ins_cost, sub_cost)
        v0, v1 = v1, v0
    return v0[-1]


def _raise_noenv_error(name: str) -> t.NoReturn:
    THRESHOLD = 4
    best = None, int(1e9)
    for env_name in _Registory.registered_envs.keys():
        dist = _levenshtein_distance(name, env_name)
        if dist <= THRESHOLD and dist < best[1]:
            best = env_name, dist
    msg = f"Enviroment name {name} is not registered"
    candidate, _ = best
    if candidate is not None:
        msg += f"\n Do you mean '{candidate}'?"
    raise ValueError(msg)


def description(env_name: str) -> str:
    env_spec = _Registory.registered_envs.get(env_class, None)
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
) -> Environment:
    env_spec = _Registory.registered_envs.get(env_name, None)
    if env_spec is None:
        _raise_noenv_error(env_name)
    return env_spec.cls(*args, **kwargs)


def register(
    name: str,
    env_class: t.Type[Environment],
    description: t.Optional[str] = None,
) -> None:
    if name in _Registory.registered_envs:
        raise ValueError(f"{name} is already registered")
    _Registory.registered_envs[name] = _EnviromentSpec(env_class, description)


register(
    "Waterworld-v0",
    WaterWorld,
    description="A simple continuous control enviroment,"
    + " where pursuer arechas aims to eat foods, avoiding poisons.",
)
