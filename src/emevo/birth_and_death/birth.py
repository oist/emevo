from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from emevo.status import Status


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
    beta: float = 0.001
    age_delay: float = 1000.0
    energy_delay: float = 8.0

    def _exp_age(self, age: float) -> float:
        return np.exp(-self.beta * (age - self.age_delay))

    def _exp_neg_energy(self, energy: float) -> float:
        return np.exp(self.energy_delay - energy)

    def asexual(self, status: Status) -> float:
        exp_neg_age = self._exp_age(status.age)
        exp_neg_energy = self._exp_neg_energy(status.energy)
        return self.scale / (1.0 + self.alpha * (exp_neg_age + exp_neg_energy))

    def sexual(self, status_a: Status, status_b: Status) -> float:
        exp_neg_age_a = self._exp_age(status_a.age)
        exp_neg_energy_a = self._exp_neg_energy(status_a.energy)
        exp_neg_age_b = self._exp_age(status_b.age)
        exp_neg_energy_b = self._exp_neg_energy(status_b.energy)
        sum_exp = exp_neg_age_a + exp_neg_energy_a + exp_neg_age_b + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)


@dataclasses.dataclass
class EnergyLogistic(BirthFunction):
    """Only energy is important to give birth."""

    scale: float
    alpha: float
    delay: float = 8.0

    def _exp_neg_energy(self, energy: float) -> float:
        return np.exp(self.delay - energy)

    def asexual(self, status: Status) -> float:
        exp_neg_energy = self._exp_neg_energy(status.energy)
        return self.scale / (1.0 + self.alpha * exp_neg_energy)

    def sexual(self, status_a: Status, status_b: Status) -> float:
        exp_neg_energy_a = self._exp_neg_energy(status_a.energy)
        exp_neg_energy_b = self._exp_neg_energy(status_b.energy)
        sum_exp = exp_neg_energy_a + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)


@dataclasses.dataclass
class EnergyLogisticMeta(BirthFunction):
    """Only energy is important to give birth."""

    scale: float
    alpha: float
    delay: float = 8.0

    def _exp_neg_energy(self, status: Status) -> float:
        assert status.metadata is not None
        energy_delay = status.metadata.get("delay", self.delay)
        return np.exp(energy_delay - status.energy)

    def asexual(self, status: Status) -> float:
        assert status.metadata is not None
        exp_neg_energy = self._exp_neg_energy(status)
        scale = status.metadata.get("scale", self.scale)
        alpha = status.metadata.get("alpha", self.alpha)
        return scale / (1.0 + alpha * exp_neg_energy)

    def sexual(self, status_a: Status, status_b: Status) -> float:
        assert status_a.metadata is not None and status_b.metadata is not None
        exp_neg_energy_a = self._exp_neg_energy(status_a)
        exp_neg_energy_b = self._exp_neg_energy(status_b)
        sum_exp = exp_neg_energy_a + exp_neg_energy_b
        scale_a = status_a.metadata.get("scale", self.scale)
        alpha_a = status_a.metadata.get("alpha", self.alpha)
        scale_b = status_b.metadata.get("scale", self.scale)
        alpha_b = status_b.metadata.get("alpha", self.alpha)
        scale = (scale_a + scale_b) / 2
        alpha = (alpha_a + alpha_b) / 2
        return scale / (1.0 + alpha * sum_exp)
