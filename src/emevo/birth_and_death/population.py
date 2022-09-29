""" Compute population statistics based on birth and hazard functions.
"""

import dataclasses

from scipy import integrate

from emevo.birth_and_death.death import HazardFunction
from emevo.birth_and_death.statuses import HasAgeAndEnergy


@dataclasses.dataclass
class _AgeAndEnergy:
    age: float
    energy: float


def cumulative_survival(
    hazard: HazardFunction[HasAgeAndEnergy],
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    survival = lambda t: hazard.survival(_AgeAndEnergy(t, energy))
    return integrate.quad(survival, 0, max_age)[0]


def stable_birth_rate(
    hazard: HazardFunction[HasAgeAndEnergy],
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    cumsuv = cumulative_survival(hazard, energy=energy, max_age=max_age)
    return 1.0 / cumsuv
