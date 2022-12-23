""" Collection of hazard functions
"""
import dataclasses
from typing import Protocol

import numpy as np

from emevo.birth_and_death.core import Status


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
        alpha = self(status)
        return alpha * status.age

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class Gompertz(Constant):
    """
    Hazard with exponentially increasing death rate.
    α = α_const + α_energy * exp(-γenergy)
    h(t) = α exp(βt)
    H(t) = α/β exp(βt)
    S(t) = exp(-α/β exp(βt))
    """

    beta: float = 1e-5

    def __call__(self, status: Status) -> float:
        return self._alpha(status) * np.exp(self.beta * status.age)

    def cumulative(self, status: Status) -> float:
        return self._alpha(status) / self.beta * np.exp(self.beta * status.age)

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))


@dataclasses.dataclass
class BEGompertz(HazardFunction):
    """Another parametrization of Gompertz. Not used now."""

    alpha: float = 2e-5
    beta_age: float = 1e-5
    beta_energy: float = 1e-5

    def _beta_sum(self, status: Status) -> float:
        return self.beta_age * status.age - self.beta_energy * status.energy

    def __call__(self, status: Status) -> float:
        return self.alpha * np.exp(self._beta_sum(status))

    def cumulative(self, status: Status) -> float:
        return self.alpha / self.beta_age * np.exp(self._beta_sum(status))

    def survival(self, status: Status) -> float:
        return np.exp(-self.cumulative(status))
