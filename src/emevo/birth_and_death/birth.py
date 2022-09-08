from __future__ import annotations

from typing import Callable

import numpy as np

from emevo import Encount
from emevo.birth_and_death.statuses import HasAgeAndEnergy, HasEnergy


def log_prod(
    scale_energy: float,
    scale_prob: float,
) -> Callable[[tuple[HasAgeAndEnergy, HasAgeAndEnergy], Encount], float]:
    def _scaled_log(value: float, scale: float) -> float:
        return np.log(1 + max(value, 0.0) * scale)

    def success_prob(
        statuses: tuple[HasAgeAndEnergy, HasAgeAndEnergy],
        _encount: Encount,
    ) -> float:
        log_e1, log_e2 = map(
            lambda status: _scaled_log(status.energy, scale_energy), statuses
        )
        return min(1.0, log_e1 * log_e2 * scale_prob)

    return success_prob


def linear_to_energy(
    t_max: float = 10000,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasEnergy], float]:
    energy_range = energy_max - energy_min

    def success_prob(status: HasEnergy) -> float:
        energy_ratio = (status.energy + energy_min) / energy_range
        return energy_ratio * 2 / t_max + 1 / t_max

    return success_prob


def linear_to_energy_sum(
    t_max: float = 10000,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasEnergy, HasEnergy], float]:
    energy_range = (energy_max - energy_min) * 2

    def success_prob(status_a: HasEnergy, status_b: HasEnergy) -> float:
        energy_sum = status_a.energy + status_b.energy
        energy_ratio = (energy_sum - energy_min) / energy_range
        return energy_ratio * 2 / t_max + 1 / t_max

    return success_prob
