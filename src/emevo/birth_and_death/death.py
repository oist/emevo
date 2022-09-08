""" Set of death functions for some statuses
"""
from typing import Callable

import numpy as np

from emevo.birth_and_death.statuses import HasAgeAndEnergy


def hunger_or_infirmity(
    energy_threshold: float,
    age_threshold: float,
) -> Callable[[HasAgeAndEnergy], float]:
    """
    A deterministic death function where an agent dies when
    - its energy level is lower than the energy thershold or
    - its age is older than the the age thershold
    """

    def death_prob_fn(status: HasAgeAndEnergy) -> float:
        if status.energy < energy_threshold or age_threshold < status.age:
            return 1.0
        else:
            return 0.0

    return death_prob_fn


def gompertz(
    n_0: float = 0.0001,
    n_max: float = 0.01,
    b_const: float = 0.0004,
    b_age: float = 0.0004,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasAgeAndEnergy], float]:
    """https://en.wikipedia.org/wiki/Gompertz_function#Gompertz_curve"""
    energy_range = energy_max - energy_min

    def death_prob_fn(status: HasAgeAndEnergy) -> float:
        energy_ratio = (status.energy - energy_min) / energy_range
        b = b_const + b_age * (1.0 - energy_ratio)
        return n_0 * np.exp(np.log(n_max / n_0) * (1.0 - np.exp(-b * status.age)))

    return death_prob_fn


def logistic(
    n_0: float = 0.0001,
    n_max: float = 0.01,
    b: float = 0.01,
    c_const: float = 0.0004,
    c_age: float = 0.0004,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasAgeAndEnergy], float]:
    """From https://journals.asm.org/doi/10.1128/aem.56.6.1875-1881.1990"""
    energy_range = energy_max - energy_min

    def death_prob_fn(status: HasAgeAndEnergy) -> float:
        energy_ratio = (status.energy - energy_min) / energy_range
        c = c_const + c_age * (1.0 - energy_ratio)
        return (n_max - n_0) / (1 + np.exp(b - c * status.age))

    return death_prob_fn
