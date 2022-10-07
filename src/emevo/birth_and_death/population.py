""" Compute population statistics based on birth and hazard functions.
"""


from scipy import integrate

from emevo.birth_and_death.death import HazardFunction
from emevo.birth_and_death.statuses import Status


def cumulative_survival(
    hazard: HazardFunction,
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    result = integrate.quad(hazard.survival(Status(age=t, energy=energy)), 0, max_age)
    return result[0]


def stable_birth_rate(
    hazard: HazardFunction,
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    cumsuv = cumulative_survival(hazard, energy=energy, max_age=max_age)
    return 1.0 / cumsuv
