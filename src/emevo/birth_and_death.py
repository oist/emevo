""" Evaluate birth and death probabilities.
"""
import dataclasses
from typing import Protocol

import jax
import jax.numpy as jnp
from scipy import integrate


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


@dataclasses.dataclass
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


@dataclasses.dataclass
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

    def _alpha(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        alpha_energy = self.alpha_energy * jnp.exp(-self.gamma * energy)
        return self.alpha_const + alpha_energy

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self._alpha(status)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self(status) * age


@dataclasses.dataclass
class EnergyLogisticHazard(HazardFunction):
    """
    Hazard with death rate that only depends on energy.
    h(e) = h_max (1 - 1 / (1 + αexp(e0 - e))
    """

    alpha: float = 1.0
    hmax: float = 1.0
    e0: float = 3.0

    def _energy_death_rate(self, energy: jax.Array) -> jax.Array:
        exp_neg_energy = self.alpha * jnp.exp(self.e0 - energy)
        return self.hmax * (1.0 - 1.0 / (1.0 + self.alpha * exp_neg_energy))

    def __call__(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self._energy_death_rate(energy)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        return self._energy_death_rate(energy) * age


@dataclasses.dataclass
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
        return self._alpha(status) * jnp.exp(self.beta * age)

    def cumulative(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        alpha = self._alpha(status)
        ht = alpha / self.beta * jnp.exp(self.beta * age)
        h0 = alpha / self.beta
        return ht - h0


@dataclasses.dataclass
class ELGompertz(EnergyLogisticHazard):
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


class BirthFunction(Protocol):
    def asexual(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        """Birth function b(t)"""
        ...

    def sexual(
        self,
        age_a: jax.Array,
        energy_a: jax.Array,
        age_b: jax.Array,
        energy_b: jax.Array,
    ) -> jax.Array:
        """Birth function b(t)"""
        ...


@dataclasses.dataclass
class LogisticBirth(BirthFunction):
    scale: float
    alpha: float = 1.0
    beta: float = 0.001
    age_delay: float = 1000.0
    energy_delay: float = 8.0

    def _exp_age(self, age: jax.Array) -> jax.Array:
        return jnp.exp(-self.beta * (age - self.age_delay))

    def _exp_neg_energy(self, energy: jax.Array) -> jax.Array:
        return jnp.exp(self.energy_delay - energy)

    def asexual(self, age: jax.Array, energy: jax.Array) -> jax.Array:
        exp_neg_age = self._exp_age(age)
        exp_neg_energy = self._exp_neg_energy(energy)
        return self.scale / (1.0 + self.alpha * (exp_neg_age + exp_neg_energy))

    def sexual(
        self,
        age_a: jax.Array,
        energy_a: jax.Array,
        age_b: jax.Array,
        energy_b: jax.Array,
    ) -> jax.Array:
        exp_neg_age_a = self._exp_age(age_a)
        exp_neg_energy_a = self._exp_neg_energy(energy_a)
        exp_neg_age_b = self._exp_age(age_b)
        exp_neg_energy_b = self._exp_neg_energy(energy_b)
        sum_exp = exp_neg_age_a + exp_neg_energy_a + exp_neg_age_b + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)


@dataclasses.dataclass
class EnergyLogisticBirth(BirthFunction):
    """
    Only energy is important to give birth.
    b(t) = scale / (1.0 + α x exp(delay - e(t)))
    """

    scale: float
    alpha: float = 1.0
    delay: float = 8.0

    def _exp_neg_energy(self, energy: jax.Array) -> jax.Array:
        return jnp.exp(self.delay - energy)

    def asexual(self, _age: jax.Array, energy: jax.Array) -> jax.Array:
        exp_neg_energy = self._exp_neg_energy(energy)
        return self.scale / (1.0 + self.alpha * exp_neg_energy)

    def sexual(
        self,
        _age_a: jax.Array,
        energy_a: jax.Array,
        _age_b: jax.Array,
        energy_b: jax.Array,
    ) -> jax.Array:
        del _age_a, _age_b
        exp_neg_energy_a = self._exp_neg_energy(energy_a)
        exp_neg_energy_b = self._exp_neg_energy(energy_b)
        sum_exp = exp_neg_energy_a + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)


def compute_cumulative_hazard(
    hazard: HazardFunction,
    *,
    energy: float = 10.0,
    max_age: float = 1e6,
) -> float:
    """Compute cumulative hazard using numeric integration"""
    energy_arr = jnp.array(energy)
    result = integrate.quad(
        lambda t: hazard(jnp.array(t), energy_arr).item(),
        0.0,
        max_age,
        limit=10000,
    )
    return result[0]


def compute_cumulative_survival(
    hazard: HazardFunction,
    *,
    energy: float = 10.0,
    max_age: float = 1e6,
) -> float:
    """Compute cumulative survival rate using numeric integration"""
    energy_arr = jnp.array(energy)
    result = integrate.quad(
        lambda t: hazard(jnp.array(t), energy_arr).item(),
        0,
        max_age,
    )
    return result[0]


def compute_stable_birth_rate(
    hazard: HazardFunction,
    *,
    energy: float = 10.0,
    max_age: float = 1e6,
) -> float:
    """Compute cumulative survival rate using numeric integration"""
    cumsuv = compute_cumulative_survival(hazard, energy=energy, max_age=max_age)
    return 1.0 / cumsuv


def expected_n_children(
    *,
    birth: BirthFunction,
    hazard: HazardFunction,
    max_age: float = 1e6,
    asexual: bool = False,
    energy: float = 10.0,
) -> float:
    energy_arr = jnp.array(energy)

    def integrated(t: float) -> float:
        age_arr = jnp.array(t)
        if asexual:
            b = birth.asexual(age_arr, energy_arr).item()
        else:
            b = birth.sexual(age_arr, energy_arr, age_arr, energy_arr).item()
        h = hazard.survival(age_arr, energy_arr).item()
        return h * b

    result = integrate.quad(integrated, 0, max_age)
    return result[0]


def evaluate_hazard(
    hf: HazardFunction,
    age_from: jax.Array,
    age_to: jax.Array,
):
    assert False, "unimplemnted"
