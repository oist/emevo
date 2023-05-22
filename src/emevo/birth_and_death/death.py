""" Collection of hazard functions
"""
import dataclasses
from typing import Protocol

import numpy as np

from emevo.status import Status


class HazardFunction(Protocol):
    def __call__(self, status: Status) -> float:
        """Hazard function h(t)"""
        ...

    def cumulative(self, status: Status) -> float:
        """Cumulative hazard function H(t) = ∫h(t)"""
        ...

    def survival(self, status: Status) -> float:
        """Survival Rate S(t) = exp(-H(t))"""
        ...


@dataclasses.dataclass
class Deterministic(HazardFunction):
    """
    A deterministic hazard function where an agent dies when
    - its energy level is lower than the energy thershold or
    - its age is older than the the age thershold
    """

    energy_threshold: float
    age_threshold: float

    def __call__(self, status: Status) -> float:
        if status.energy < self.energy_threshold or self.age_threshold < status.age:
            return 1.0
        else:
            return 0.0

    def cumulative(self, status: Status) -> float:
        return self(status)

    def survival(self, status: Status) -> float:
        if status.energy < self.energy_threshold or self.age_threshold < status.age:
            return 0.0
        else:
            return 1.0


@dataclasses.dataclass
class Constant(HazardFunction):
    """
    Hazard with constant death rate.
    Energy
    α = α_const + α_energy * exp(-γenergy)
    h(t) = α
    H(t) = αt
    S(t) = exp(-αt)
    """

    alpha_const: float = 1e-5
    alpha_energy: float = 1e-6
    gamma: float = 1.0

    def _alpha(self, status: Status) -> float:
        alpha_energy = self.alpha_energy * np.exp(-self.gamma * status.energy)
        return self.alpha_const + alpha_energy

    def __call__(self, status: Status) -> float:
        return self._alpha(status)

    def cumulative(self, status: Status) -> float:
        return self(status) * status.age

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class EnergyLogistic(HazardFunction):
    """
    Hazard with death rate that only depends on energy.
    h(e) = h_max (1 - 1 / (1 + αexp(e0 - e))
    """

    alpha: float = 1.0
    hmax: float = 1.0
    e0: float = 3.0

    def _energy_death_rate(self, energy: float) -> float:
        exp_neg_energy = self.alpha * np.exp(self.e0 - energy)
        return self.hmax * (1.0 - 1.0 / (1.0 + self.alpha * exp_neg_energy))

    def __call__(self, status: Status) -> float:
        return self._energy_death_rate(status.energy)

    def cumulative(self, status: Status) -> float:
        return self._energy_death_rate(status.energy) * status.age

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class Gompertz(Constant):
    """
    Hazard with exponentially increasing death rate.
    α = α_const + α_energy * exp(-γenergy)
    h(t) = α exp(βt)
    H(t) = α/β exp(βt)
    S(t) = exp(-H(t))
    """

    beta: float = 1e-5

    def __call__(self, status: Status) -> float:
        return self._alpha(status) * np.exp(self.beta * status.age)

    def cumulative(self, status: Status) -> float:
        alpha = self._alpha(status)
        ht = alpha / self.beta * np.exp(self.beta * status.age)
        h0 = alpha / self.beta
        return ht - h0

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class SeparatedGompertz(EnergyLogistic):
    """
    Hazard with exponentially increasing death rate.
    h(e) = -scale / (1 + αexp(d - e))
    h(t) = αexp(βt) + h(e)
    H(t) = α/β exp(βt) + h(e)t
    S(t) = exp(-H(t))
    """

    alpha_age: float = 1e-6
    beta: float = 1e-5

    def __call__(self, status: Status) -> float:
        age = self.alpha_age * np.exp(self.beta * status.age)
        energy = self._energy_death_rate(status.energy)
        return age + energy

    def cumulative(self, status: Status) -> float:
        energy = self._energy_death_rate(status.energy) * status.age
        ht = energy + self.alpha_age / self.beta * np.exp(self.beta * status.age)
        h0 = self.alpha_age / self.beta
        return ht - h0

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class SimplifiedGompertz(HazardFunction):
    """
    Similar to SeparatedGompertz, but with less parameters.
    h(e) = αexp(-βe)
    h(t) = αexp(βt) + h(e)
    H(t) = α/β exp(βt) + h(e)t
    S(t) = exp(-H(t))
    """

    alpha_e: float = 0.01
    alpha_t: float = 1e-4
    beta_e: float = 0.8
    beta_t: float = 1e-5

    def _he(self, energy: float) -> float:
        return self.alpha_e * np.exp(-self.beta_e * energy)

    def __call__(self, status: Status) -> float:
        age = self.alpha_t * np.exp(self.beta_t * status.age)
        energy = self._he(status.energy)
        return age + energy

    def cumulative(self, status: Status) -> float:
        energy = self._he(status.energy) * status.age
        ht = energy + self.alpha_t / self.beta_t * np.exp(self.beta_t * status.age)
        h0 = self.alpha_t / self.beta_t
        return ht - h0

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))
