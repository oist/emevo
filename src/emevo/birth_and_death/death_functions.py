""" Set of death functions for some statuses
"""
import typing as t

import numpy as np

from .statuses import AgeAndEnergy


def hunger_or_infirmity(
    energy_threshold: float,
    age_threshold: float,
) -> t.Callable[[AgeAndEnergy], float]:
    def death_prob_fn(status: AgeAndEnergy) -> bool:
        if status.energy < energy_threshold or age_threshold < status.age:
            return 1.0
        else:
            return 0.0

    return death_prob_fn


def energy_to_gompertz_r_fn(
    energy_min: float,
    energy_max: float,
    base: float = 0.001,
) -> t.Callable[[float], float]:
    energy_range = energy_max - energy_min

    def energy_to_gompertz_r(energy: float) -> float:
        normalized = max(0.0, energy - energy_min) / energy_range
        return normalized * base

    return energy_to_gompertz_r


def gompertz_hazard(
    energy_threshold: float,x
    energy_to_gompertz_r: t.Callable[[float], float] = energy_to_gompertz_r_fn(
        -5.0, 15.0
    ),
    gompertz_alpha: float = 0.001,
) -> t.Callable[[AgeAndEnergy], float]:
    """Gompertz hazard function is defined by λ(x) = R exp(αx)"""

    def death_prob_fn(status: AgeAndEnergy) -> bool:
        if status.energy < energy_threshold:
            return 1.0
        r = energy_to_gompertz_r(status.energy)
        hazard = np.exp(gompertz_alpha * status.age)
        return min(r * hazard, 1.0)

    return death_prob_fn
