from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from emevo.birth_and_death.core import Status


class BirthFunction(Protocol):
    def asexual(self, status: Status) -> float:
        """Birth function b(t)"""
        ...

    def sexual(self, status_a: Status, status_b: Status) -> float:
        """Birth function b(t)"""
        ...


@dataclasses.dataclass
class Logistic(BirthFunction):
    scale: float
    alpha: float
    beta_age: float = 0.001
    beta_energy: float = 1.0
    age_delay: float = 1000.0
    energy_delay: float = 8.0

    def _exp_neg_age(self, age: float) -> float:
        return np.exp(-self.beta_age * (age - self.age_delay))

    def _exp_neg_energy(self, energy: float) -> float:
        return np.exp(-self.beta_energy * (energy - self.energy_delay))

    def asexual(self, status: Status) -> float:
        exp_neg_age = self._exp_neg_age(status.age)
        exp_neg_energy = self._exp_neg_energy(status.energy)
        return self.scale / (1.0 + self.alpha * (exp_neg_age + exp_neg_energy))

    def sexual(self, status_a: Status, status_b: Status) -> float:
        exp_neg_age_a = self._exp_neg_age(status_a.age)
        exp_neg_energy_a = self._exp_neg_energy(status_a.energy)
        exp_neg_age_b = self._exp_neg_age(status_b.age)
        exp_neg_energy_b = self._exp_neg_energy(status_b.energy)
        sum_exp = exp_neg_age_a + exp_neg_energy_a + exp_neg_age_b + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)
