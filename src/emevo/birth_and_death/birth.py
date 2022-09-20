from __future__ import annotations

from typing import Callable

from emevo.birth_and_death.statuses import HasEnergy


def linear_to_energy(
    alpha1: float = 0.01,
    alpha2: float = 0.005,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasEnergy], float]:
    """α1 + α2 * energy"""
    energy_range = energy_max - energy_min

    def success_prob(status: HasEnergy) -> float:
        energy_ratio = (status.energy + energy_min) / energy_range
        return alpha1 + energy_ratio * alpha2

    return success_prob


def linear_to_energy_sum(
    alpha1: float = 0.01,
    alpha2: float = 0.005,
    energy_min: float = -5.0,
    energy_max: float = 15.0,
) -> Callable[[HasEnergy, HasEnergy], float]:
    """α1 + α2 * (parent1.energy + parent2.energy)"""
    energy_range = (energy_max - energy_min) * 2

    def success_prob(status_a: HasEnergy, status_b: HasEnergy) -> float:
        energy_sum = status_a.energy + status_b.energy
        energy_ratio = (energy_sum - energy_min) / energy_range
        return alpha1 + energy_ratio * alpha2

    return success_prob
