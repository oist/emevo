""" Collection of hazard functions
"""
import dataclasses
from typing import Protocol, TypeVar

import numpy as np

from emevo.birth_and_death.statuses import HasAgeAndEnergy

# typevar for status
S = TypeVar("S", contravariant=True)


class HazardFunction(Protocol[S]):
    def __call__(self, status: S) -> float:
        """Hazard function h(t)"""
        ...

    def cumulative(self, status: S) -> float:
        """Cumulative hazard function H(t) = ∫h(t)"""
        ...

    def survival(self, status: S) -> float:
        """Survival Rate S(t) = exp(-H(t))"""
        ...


class _EnergyRatio:
    energy_min: float
    energy_range: float

    def _energy_ratio(self, energy: float) -> float:
        return (energy - self.energy_min) / self.energy_range

    @property
    def energy_mean(self) -> float:
        return self.energy_min + self.energy_range * 0.5


@dataclasses.dataclass
class Deterministic(HazardFunction[HasAgeAndEnergy]):
    """
    A deterministic hazard function where an agent dies when
    - its energy level is lower than the energy thershold or
    - its age is older than the the age thershold
    """

    energy_threshold: float
    age_threshold: float

    def __call__(self, status: HasAgeAndEnergy) -> float:
        if status.energy < self.energy_threshold or self.age_threshold < status.age:
            return 1.0
        else:
            return 0.0

    def cumulative(self, status: HasAgeAndEnergy) -> float:
        return self(status)

    def survival(self, status: HasAgeAndEnergy) -> float:
        if status.energy < self.energy_threshold or self.age_threshold < status.age:
            return 0.0
        else:
            return 1.0


@dataclasses.dataclass
class Constant(HazardFunction[HasAgeAndEnergy], _EnergyRatio):
    """
    Hazard with constant death rate.
    Same as Weibull with β == 1.
    α = α1 + α2 * (1.0 - energy_ratio)
    h(t) = α
    H(t) = αt
    S(t) = exp(-αt)
    """

    alpha1: float = 4e-5
    alpha2: float = 4e-5
    energy_min: float = -5.0
    energy_max: float = 15.0
    energy_range: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.energy_range = self.energy_max - self.energy_min

    def __call__(self, status: HasAgeAndEnergy) -> float:
        energy_ratio = self._energy_ratio(status.energy)
        return self.alpha1 + self.alpha2 * (1.0 - energy_ratio)

    def cumulative(self, status: HasAgeAndEnergy) -> float:
        alpha = self(status)
        return alpha * status.age

    def survival(self, status: HasAgeAndEnergy) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class Fractional(HazardFunction[HasAgeAndEnergy], _EnergyRatio):
    """
    Hazard that suddenly decreases by t/(αt + β).
    α = α1 + α2 * (1.0 - energy_ratio)
    h(t) = t/(αt + β)
    H(t) = log(αt + β)
    S(t) = 1/(αt + β)
    """

    alpha1: float = 4e-5
    alpha2: float = 4e-5
    beta: float = 1.1
    energy_min: float = -5.0
    energy_max: float = 15.0
    energy_range: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.energy_range = self.energy_max - self.energy_min

    def _alpha(self, energy: float) -> float:
        energy_ratio = self._energy_ratio(energy)
        return self.alpha1 + self.alpha2 * (1.0 - energy_ratio)

    def __call__(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return status.age / (status.age * alpha + self.beta)

    def cumulative(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return status.age * alpha + self.beta

    def survival(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return 1.0 / (status.age * alpha + self.beta)


@dataclasses.dataclass
class Gompertz(HazardFunction[HasAgeAndEnergy], _EnergyRatio):
    """
    Hazard with increasing/decreasing death rate with a constant rate.
    α = α1 + α2 * (1.0 - energy_ratio)
    h(t) = α exp(βt)
    H(t) = α/β exp(βt)
    S(t) = exp(-α/β exp(βt))
    """

    alpha1: float = 2e-5
    alpha2: float = 2e-5
    beta: float = 1e-5
    energy_min: float = -5.0
    energy_max: float = 15.0
    energy_range: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.energy_range = self.energy_max - self.energy_min

    def _alpha(self, energy: float) -> float:
        energy_ratio = self._energy_ratio(energy)
        return self.alpha1 + self.alpha2 * (1.0 - energy_ratio)

    def __call__(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return alpha * np.exp(self.beta * status.age)

    def cumulative(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return (alpha / self.beta) * np.exp(self.beta * status.age)

    def survival(self, status: HasAgeAndEnergy) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class Weibull(HazardFunction[HasAgeAndEnergy], _EnergyRatio):
    """
    Hazard with constantly increasing/decreasing death rate.
    If β == 1, this hazard is the same as Constant.
    α = α1 + α2 * (1.0 - energy_ratio)
    h(t) = βα^βt^(β - 1)
    H(t) = (αt)^β
    S(t) = exp(-(αt)^β)
    """

    alpha1: float = 4e-5
    alpha2: float = 4e-5
    beta: float = 1.1
    energy_min: float = -5.0
    energy_max: float = 15.0
    energy_range: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.energy_range = self.energy_max - self.energy_min

    def _alpha(self, energy: float) -> float:
        energy_ratio = self._energy_ratio(energy)
        return self.alpha1 + self.alpha2 * (1.0 - energy_ratio)

    def __call__(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return self.beta * (alpha**self.beta) * (status.age ** (self.beta - 1.0))

    def cumulative(self, status: HasAgeAndEnergy) -> float:
        alpha = self._alpha(status.energy)
        return (alpha * status.age) ** self.beta

    def survival(self, status: HasAgeAndEnergy) -> float:
        return np.exp(-self.cumulative(status))
