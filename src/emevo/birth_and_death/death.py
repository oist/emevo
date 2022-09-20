""" Collection of hazard functions
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

    def hazard_fn(status: HasAgeAndEnergy) -> float:
        if status.energy < energy_threshold or age_threshold < status.age:
            return 1.0
        else:
            return 0.0

    return hazard_fn


def gompertz(
    alpha1: float = 1e-5,
    alpha2: float = 1e-5,
    beta: float = 1e-4,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasAgeAndEnergy], float]:
    """
    Gompertz hazard function h(t) = (α1 + α2 * (1 - energy)) * exp(age * β).
    Reference: https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality.
    """
    energy_range = energy_max - energy_min

    def hazard_fn(status: HasAgeAndEnergy) -> float:
        energy_ratio = (status.energy - energy_min) / energy_range
        alpha = alpha1 + alpha2 * (1.0 - energy_ratio)
        return alpha * np.exp(status.age * beta)

    return hazard_fn


def weibull(
    alpha1: float = 4e-5,
    alpha2: float = 4e-5,
    beta: float = 1.1,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasAgeAndEnergy], float]:
    """
    Weibull hazard function: h(t) = β (α1 + α2(1 - energy))^(β) age^(β - 1)
    https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality
    """
    energy_range = energy_max - energy_min

    def hazard_fn(status: HasAgeAndEnergy) -> float:
        energy_ratio = (status.energy - energy_min) / energy_range
        alpha = alpha1 + alpha2 * (1.0 - energy_ratio)
        return beta * (alpha**beta) * (status.age ** (beta - 1.0))

    return hazard_fn
