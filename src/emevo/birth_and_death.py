""" Evaluate birth and death probabilities.
"""

import dataclasses
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid


class HazardFunction(Protocol):
    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Hazard function h(t)"""
        ...

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Cumulative hazard function H(t) = ∫h(t)"""
        ...

    def survival(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Survival Rate S(t) = exp(-H(t))"""
        return jnp.exp(-self.cumulative(age, energy))


@dataclasses.dataclass(frozen=True)
class DeterministicHazard(HazardFunction):
    """
    A deterministic hazard function where an agent dies when
    - its energy level is lower than the energy thershold or
    - its age is older than the the age thershold
    """

    energy_threshold: float
    age_threshold: float

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        res = jnp.logical_or(
            energy < self.energy_threshold,
            self.age_threshold < age,
        )
        return res.astype(jnp.float32)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return jnp.where(
            energy < self.energy_threshold,
            age,
            jnp.where(
                self.age_threshold < age,
                age - self.age_threshold,
                0.0,
            ),
        )


@dataclasses.dataclass(frozen=True)
class ConstantHazard(HazardFunction):
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

    def _alpha(self, _age: jax.Array, energy: jax.Array) -> jax.Array:
        del _age
        alpha_energy = self.alpha_energy * jnp.exp(-self.gamma * energy)
        return self.alpha_const + alpha_energy

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self._alpha(age, energy)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self(age, energy) * age


@dataclasses.dataclass(frozen=True)
class EnergyLogisticHazard(HazardFunction):
    """
    Hazard with death rate that only depends on energy.
    h(e) = h_max (1 - 1 / (1 + αexp(e0 - e))
    """

    alpha: float = 1.0
    scale: float = 1.0
    e0: float = 3.0

    def _energy_death_rate(self, energy: jax.Array) -> jax.Array:
        return self.scale * (1.0 - 1.0 / (1.0 + self.alpha * jnp.exp(self.e0 - energy)))

    def __call__(self, _age: jax.Array, energy: jax.Array) -> jax.Array:
        del _age
        return self._energy_death_rate(energy)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self._energy_death_rate(energy) * age


@dataclasses.dataclass(frozen=True)
class GompertzHazard(ConstantHazard):
    """
    Hazard with exponentially increasing death rate.
    α = α_const + α_energy * exp(-γenergy)
    h(t) = α exp(βt)
    H(t) = α/β exp(βt)
    S(t) = exp(-H(t))
    """

    beta: float = 1e-5

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self._alpha(age, energy) * jnp.exp(self.beta * age)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        alpha = self._alpha(age, energy)
        ht = alpha / self.beta * jnp.exp(self.beta * age)
        h0 = alpha / self.beta
        return ht - h0


@dataclasses.dataclass(frozen=True)
class ELGompertzHazard(EnergyLogisticHazard):
    """
    Exponentially increasing with time + EnergyLogistic
    h(e) = h_max (1 - 1 / (1 + αexp(e0 - e))
    h(t) = αexp(βt) + h(e)
    H(t) = α/β exp(βt) + h(e)t
    S(t) = exp(-H(t))
    """

    alpha_age: float = 1e-6
    beta: float = 1e-5

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        age = self.alpha_age * jnp.exp(self.beta * age)
        energy = self._energy_death_rate(energy)
        return age + energy

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        energy = self._energy_death_rate(energy) * age
        ht = energy + self.alpha_age / self.beta * jnp.exp(self.beta * age)
        h0 = self.alpha_age / self.beta
        return ht - h0


@dataclasses.dataclass(frozen=True)
class SlopeELGHazard(EnergyLogisticHazard):
    alpha: float = 0.1
    scale: float = 1.0
    slope: float = 0.1
    alpha_age: float = 1e-6
    beta: float = 1e-5

    def _energy_death_rate(self, energy: jax.Array) -> jax.Array:
        return self.scale * (
            1.0 - 1.0 / (1.0 + self.alpha * jnp.exp(-energy * self.slope))
        )

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        age = self.alpha_age * jnp.exp(self.beta * age)
        energy = self._energy_death_rate(energy)
        return age + energy

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        energy = self._energy_death_rate(energy) * age
        ht = energy + self.alpha_age / self.beta * jnp.exp(self.beta * age)
        h0 = self.alpha_age / self.beta
        return ht - h0


class BirthFunction(Protocol):
    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Birth function b(t)"""
        ...

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Cumulative birth function B(t) = ∫b(t)"""
        ...


@dataclasses.dataclass(frozen=True)
class EnergyLogisticBirth(BirthFunction):
    """
    Only energy is important to give birth.
    b(t) = scale / (1.0 + α x exp(delay - e(t)))
    """

    alpha: float = 1.0
    scale: float = 0.1
    e0: float = 8.0

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        del age
        return self.scale / (1.0 + self.alpha * jnp.exp(self.e0 - energy))

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Birth function b(t)"""
        return age * self(age, energy)


@dataclasses.dataclass(frozen=True)
class SlopeELBirth(BirthFunction):
    slope: float = 1.0
    scale: float = 0.1

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        del age
        return self.scale / (1.0 + jnp.exp(-energy * self.slope))

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Birth function b(t)"""
        return age * self(age, energy)


def compute_cumulative_hazard(
    hazard: HazardFunction,
    *,
    energy: float = 10.0,
    max_age: float = 1e6,
    n: int = 100000,
) -> float:
    """Compute cumulative hazard using numeric integration"""
    age = jnp.linspace(0.0, max_age, n)
    return trapezoid(y=hazard(age, jnp.ones(n) * energy), x=age)


def compute_cumulative_survival(
    hazard: HazardFunction,
    *,
    energy: float = 10.0,
    max_age: float = 1e6,
    n: int = 100000,
) -> float:
    """Compute cumulative survival rate using numeric integration"""
    age = jnp.linspace(0.0, max_age, n)
    return trapezoid(y=hazard.survival(age, jnp.ones(n) * energy), x=age)


def compute_stable_birth_rate(
    hazard: HazardFunction,
    *,
    energy: float = 10.0,
    max_age: float = 1e6,
    n: int = 100000,
) -> float:
    """Compute cumulative survival rate using numeric integration"""
    cumsuv = compute_cumulative_survival(hazard, energy=energy, max_age=max_age, n=n)
    return 1.0 / cumsuv


def compute_expected_n_children(
    *,
    birth: BirthFunction,
    hazard: HazardFunction,
    max_age: float = 1e6,
    energy: float = 10.0,
    n: int = 100000,
) -> float:
    age = jnp.linspace(0.0, max_age, n)
    energy_arr = jnp.ones(n) * energy
    s = hazard.survival(age, energy_arr)
    b = birth(age, energy_arr)
    return trapezoid(y=s * b, x=age)
