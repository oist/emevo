""" Set of death functions for some statuses
"""
import typing as t

import numpy as np

from .statuses import AgeAndEnergy


def hunger_or_infirmity(
    energy_threshold: float,
    age_threshold: float,
) -> t.Callable[[AgeAndEnergy], float]:
    """A completely discretized death function"""

    def death_prob_fn(status: AgeAndEnergy) -> bool:
        if status.energy < energy_threshold or age_threshold < status.age:
            return 1.0
        else:
            return 0.0

    return death_prob_fn


def gompertz_hazard(
    energy_threshold: float,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
    gompertz_alpha: float = 1e-4,
) -> t.Callable[[AgeAndEnergy], float]:
    """Gompertz hazard function is defined by λ(x) = R exp(αx)"""
    energy_range = energy_max - energy_min

    def death_prob_fn(status: AgeAndEnergy) -> bool:
        if status.energy < energy_threshold:
            return 1.0
        r = max(0.0, status.energy - energy_min) / energy_range
        hazard = np.exp(gompertz_alpha * status.age)
        return min(r * hazard, 1.0)

    return death_prob_fn
