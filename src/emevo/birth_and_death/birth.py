from __future__ import annotations

from typing import Callable

from emevo.birth_and_death.statuses import HasEnergy


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
