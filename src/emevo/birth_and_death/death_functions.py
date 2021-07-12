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


def energy_to_gompertz_r(
    energy_min: float,
    energy_max: float,
    base: float = 0.001,
) -> t.Callable[[float], float]:
    energy_range = energy_max - energy_min

    def convert_fn(energy: float) -> float:
        normalized = max(0.0, energy - energy_min) / energy_range
        return normalized * base

    return convert_fn


def gompertz_hazard(
    energy_threshold: float,
    energy_to_gompertz_r: t.Callable[[float], float] = energy_to_gompertz_r(-5.0, 15.0),
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
