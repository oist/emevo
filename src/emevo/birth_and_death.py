""" Evaluate birth and death probabilities.
"""
import dataclasses
from typing import Protocol

import jax
import jax.numpy as jnp
from scipy import integrate

from emevo.status import Status


class HazardFunction(Protocol):
    def __call__(self, status: Status) -> jax.Array:
        """Hazard function h(t)"""
        ...

    def cumulative(self, status: Status) -> jax.Array:
        """Cumulative hazard function H(t) = ∫h(t)"""
        ...

    def survival(self, status: Status) -> jax.Array:
        """Survival Rate S(t) = exp(-H(t))"""
        return jnp.exp(-self.cumulative(status))


@dataclasses.dataclass
class Deterministic(HazardFunction):
    """
    A deterministic hazard function where an agent dies when
    - its energy level is lower than the energy thershold or
    - its age is older than the the age thershold
    """

    energy_threshold: float
    age_threshold: float

    def __call__(self, status: Status) -> jax.Array:
        res = jnp.logical_or(
            status.energy < self.energy_threshold,
            self.age_threshold < status.age,
        )
        return res.astype(jnp.float32)

    def cumulative(self, status: Status) -> jax.Array:
        return jnp.where(
            status.energy < self.energy_threshold,
            status.age,
            jnp.where(
                self.age_threshold < status.age,
                status.age - self.age_threshold,
                0.0,
            ),
        )


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

    def _alpha(self, status: Status) -> jax.Array:
        alpha_energy = self.alpha_energy * jnp.exp(-self.gamma * status.energy)
        return self.alpha_const + alpha_energy

    def __call__(self, status: Status) -> jax.Array:
        return self._alpha(status)

    def cumulative(self, status: Status) -> jax.Array:
        return self(status) * status.age


@dataclasses.dataclass
class EnergyLogistic(HazardFunction):
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

    def __call__(self, status: Status) -> jax.Array:
        return self._energy_death_rate(status.energy)

    def cumulative(self, status: Status) -> jax.Array:
        return self._energy_death_rate(status.energy) * status.age


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

    def __call__(self, status: Status) -> jax.Array:
        return self._alpha(status) * jnp.exp(self.beta * status.age)

    def cumulative(self, status: Status) -> jax.Array:
        alpha = self._alpha(status)
        ht = alpha / self.beta * jnp.exp(self.beta * status.age)
        h0 = alpha / self.beta
        return ht - h0


@dataclasses.dataclass
class EnergyLogGompertz(EnergyLogistic):
    """
    Exponentially increasing with time + EnergyLogistic
    h(e) = h_max (1 - 1 / (1 + αexp(e0 - e))
    h(t) = αexp(βt) + h(e)
    H(t) = α/β exp(βt) + h(e)t
    S(t) = exp(-H(t))
    """

    alpha_age: float = 1e-6
    beta: float = 1e-5

    def __call__(self, status: Status) -> jax.Array:
        age = self.alpha_age * jnp.exp(self.beta * status.age)
        energy = self._energy_death_rate(status.energy)
        return age + energy

    def cumulative(self, status: Status) -> jax.Array:
        energy = self._energy_death_rate(status.energy) * status.age
        ht = energy + self.alpha_age / self.beta * jnp.exp(self.beta * status.age)
        h0 = self.alpha_age / self.beta
        return ht - h0


def cumulative_hazard(
    hazard: HazardFunction,
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    status = Status(
        age=jnp.array(0),
        energy=jnp.array(energy),
        is_alive=jnp.array(True),
    )
    result = integrate.quad(
        lambda t: hazard(status.replace(t=jnp.array(t))),
        0.0,
        max_age,
        limit=10000,
    )
    return result[0]


class BirthFunction(Protocol):
    def asexual(self, status: Status) -> jax.Array:
        """Birth function b(t)"""
        ...

    def sexual(self, status_a: Status, status_b: Status) -> jax.Array:
        """Birth function b(t)"""
        ...


@dataclasses.dataclass
class Logistic(BirthFunction):
    scale: float
    alpha: float = 1.0
    beta: float = 0.001
    age_delay: float = 1000.0
    energy_delay: float = 8.0

    def _exp_age(self, age: jax.Array) -> jax.Array:
        return jnp.exp(-self.beta * (age - self.age_delay))

    def _exp_neg_energy(self, energy: float) -> jax.Array:
        return jnp.exp(self.energy_delay - energy)

    def asexual(self, status: Status) -> jax.Array:
        exp_neg_age = self._exp_age(status.age)
        exp_neg_energy = self._exp_neg_energy(status.energy)
        return self.scale / (1.0 + self.alpha * (exp_neg_age + exp_neg_energy))

    def sexual(self, status_a: Status, status_b: Status) -> jax.Array:
        exp_neg_age_a = self._exp_age(status_a.age)
        exp_neg_energy_a = self._exp_neg_energy(status_a.energy)
        exp_neg_age_b = self._exp_age(status_b.age)
        exp_neg_energy_b = self._exp_neg_energy(status_b.energy)
        sum_exp = exp_neg_age_a + exp_neg_energy_a + exp_neg_age_b + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)


@dataclasses.dataclass
class EnergyLogistic(BirthFunction):
    """
    Only energy is important to give birth.
    b(t) = scale / (1.0 + α x exp(delay - e(t)))
    """

    scale: float
    alpha: float = 1.0
    delay: float = 8.0

    def _exp_neg_energy(self, energy: jax.Array) -> jax.Array:
        return jnp.exp(self.delay - energy)

    def asexual(self, status: Status) -> jax.Array:
        exp_neg_energy = self._exp_neg_energy(status.energy)
        return self.scale / (1.0 + self.alpha * exp_neg_energy)

    def sexual(self, status_a: Status, status_b: Status) -> jax.Array:
        exp_neg_energy_a = self._exp_neg_energy(status_a.energy)
        exp_neg_energy_b = self._exp_neg_energy(status_b.energy)
        sum_exp = exp_neg_energy_a + exp_neg_energy_b
        return self.scale / (1.0 + self.alpha * sum_exp)


@dataclasses.dataclass
class EnergyLogisticMeta(BirthFunction):
    """
    Only energy is important to give birth.
    Note that all fields in metadata should have 'birth_' prefix.
    """

    scale: float
    alpha: float = 1.0
    delay: float = 8.0

    def _exp_neg_energy(self, status: Status) -> jax.Array:
        assert status.metadata is not None
        energy_delay = status.metadata.get("birth_delay", self.delay)
        return jnp.exp(energy_delay - status.energy)

    def asexual(self, status: Status) -> jax.Array:
        assert status.metadata is not None
        exp_neg_energy = self._exp_neg_energy(status)
        scale = status.metadata.get("birth_scale", self.scale)
        alpha = status.metadata.get("birth_alpha", self.alpha)
        return scale / (1.0 + alpha * exp_neg_energy)

    def sexual(self, status_a: Status, status_b: Status) -> jax.Array:
        assert status_a.metadata is not None and status_b.metadata is not None
        exp_neg_energy_a = self._exp_neg_energy(status_a)
        exp_neg_energy_b = self._exp_neg_energy(status_b)
        sum_exp = exp_neg_energy_a + exp_neg_energy_b
        scale_a = status_a.metadata.get("birth_scale", self.scale)
        alpha_a = status_a.metadata.get("birth_alpha", self.alpha)
        scale_b = status_b.metadata.get("birth_scale", self.scale)
        alpha_b = status_b.metadata.get("birth_alpha", self.alpha)
        scale = (scale_a + scale_b) / 2
        alpha = (alpha_a + alpha_b) / 2
        return scale / (1.0 + alpha * sum_exp)


def cumulative_survival(
    hazard: HazardFunction,
    *,
    energy: float = 0.0,
    max_age: float = 1e6,
) -> float:
    status = Status(
        age=jnp.array(0),
        energy=jnp.array(energy),
        is_alive=jnp.array(True),
    )
    result = integrate.quad(
        lambda t: hazard.survival(status.replace(t=jnp.array(t))).item(),
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
    max_age: float = 1e6,
    asexual: bool = False,
    **status_kwargs,
) -> float:
    def integrated(t: int) -> float:
        status = Status(age=t, **status_kwargs)
        if asexual:
            b = birth.asexual(status)
        else:
            b = birth.sexual(status, status)
        h = hazard.survival(status).item()
        return h * b

    result = integrate.quad(integrated, 0, max_age)
    return result[0]
