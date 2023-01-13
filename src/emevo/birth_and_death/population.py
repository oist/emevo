""" Compute population statistics based on birth and hazard functions.
"""


from scipy import integrate

from emevo.birth_and_death.birth import BirthFunction
from emevo.birth_and_death.death import HazardFunction
from emevo.status import Status


def cumulative_survival(
    hazard: HazardFunction,
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    result = integrate.quad(
        lambda t: hazard.survival(Status(age=t, energy=energy)),
        0,
        max_age,
    )
    return result[0]


def stable_birth_rate(
    hazard: HazardFunction,
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    cumsuv = cumulative_survival(hazard, energy=energy, max_age=max_age)
    return 1.0 / cumsuv


def expected_n_children(
    *,
    birth: BirthFunction,
    hazard: HazardFunction,
    energy: float = 0.0,
    max_age: float = 1e6,
    asexual: bool = False,
) -> float:
    def integrated(t: int) -> float:
        status = Status(age=t, energy=energy)
        if asexual:
            b = birth.asexual(status)
        else:
            b = birth.sexual(status, status)
        h = hazard.survival(status)
        return h * b

    result = integrate.quad(integrated, 0, max_age)
    return result[0]
