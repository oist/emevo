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
    energy_delay: float = 8.0

    def _exp_neg_energy(self, energy: float) -> float:
        return np.exp(self.energy_delay - energy)

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

    def _exp_neg_energy(self, status: Status) -> float:
        assert status.metadata is not None
        return np.exp(status.metadata["energy_delay"] - status.energy)

    def asexual(self, status: Status) -> float:
        assert status.metadata is not None
        exp_neg_energy = self._exp_neg_energy(status)
        return status.metadata["scale"] / (
            1.0 + status.metadata["alpha"] * exp_neg_energy
        )

    def sexual(self, status_a: Status, status_b: Status) -> float:
        assert status_a.metadata is not None and status_b.metadata is not None
        exp_neg_energy_a = self._exp_neg_energy(status_a)
        exp_neg_energy_b = self._exp_neg_energy(status_b)
        sum_exp = exp_neg_energy_a + exp_neg_energy_b
        scale = (status_a.metadata["scale"] + status_b.metadata["scale"]) / 2
        alpha = (status_a.metadata["alpha"] + status_b.metadata["alpha"]) / 2
        return scale / (1.0 + alpha * sum_exp)
