from __future__ import annotations

from typing import Callable

import numpy as np

from emevo import Encount
from emevo.birth_and_death.statuses import HasAgeAndEnergy, HasEnergy


def _scaled_log(value: float, scale: float) -> float:
    return np.log(1 + max(value, 0.0) * scale)


def log_prod(
    scale_energy: float,
    scale_prob: float,
) -> Callable[[tuple[HasAgeAndEnergy, HasAgeAndEnergy], Encount], float]:
    def success_prob(
        statuses: tuple[HasAgeAndEnergy, HasAgeAndEnergy],
        _encount: Encount,
    ) -> float:
        log_e1, log_e2 = map(
            lambda status: _scaled_log(status.energy, scale_energy), statuses
        )
        return min(1.0, log_e1 * log_e2 * scale_prob)

    return success_prob


def log(scale_energy: float, scale_prob: float) -> Callable[[HasEnergy], float]:
    def success_prob(status: HasEnergy) -> float:
        return min(1.0, _scaled_log(status.energy, scale_energy) * scale_prob)

    return success_prob


def normal(
    mean: float,
    stddev: float,
    energy_max: float = 8.0,
) -> Callable[[HasAgeAndEnergy], float]:
    from scipy.stats import norm

    def success_prob(status: HasAgeAndEnergy) -> float:
        energy_coef = max(0.0, status.energy) / energy_max
        return norm.pdf(status.age, loc=mean, scale=stddev) * energy_coef

    return success_prob
