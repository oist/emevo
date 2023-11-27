import chex
import jax.numpy as jnp

import emevo.birth_and_death as bd


def test_det_hazard() -> None:
    hf = bd.DeterministicHazard(10.0, 100.0)
    age = jnp.array([10.0, 110.0, 10.0, 110.0])
    energy = jnp.array([0.0, 0.0, 20.0, 20.0])
    hazard = hf(age, energy)
    chex.assert_trees_all_close(hazard, jnp.array([1.0, 1.0, 0.0, 1.0]))


def test_constant_hazard() -> None:
    hf = bd.ConstantHazard(alpha_const=1e-5, alpha_energy=1e-6, gamma=1.0)
    age = jnp.array([10.0, 110.0, 10.0, 110.0])
    energy = jnp.array([0.0, 0.0, 20.0, 20.0])
    hazard = hf(age, energy)
    chex.assert_trees_all_close(
        hazard,
        jnp.array(
            [
                1e-5 + 1e-6 * jnp.exp(0.0),
                1e-5 + 1e-6 * jnp.exp(0.0),
                1e-5 + 1e-6 * jnp.exp(-20.0),
                1e-5 + 1e-6 * jnp.exp(-20.0),
            ]
        ),
    )


def test_energylogistic_hazard() -> None:
    hf = bd.EnergyLogisticHazard(alpha=1.0, scale=1.0, e0=3.0)
    age = jnp.array([10.0, 110.0, 10.0, 110.0])
    energy = jnp.array([0.0, 0.0, 20.0, 20.0])
    hazard = hf(age, energy)
    chex.assert_trees_all_close(
        hazard,
        jnp.array(
            [
                1.0 - (1.0 / (1.0 + jnp.exp(3.0))),
                1.0 - (1.0 / (1.0 + jnp.exp(3.0))),
                1.0 - (1.0 / (1.0 + jnp.exp(-17.0))),
                1.0 - (1.0 / (1.0 + jnp.exp(-17.0))),
            ]
        ),
    )


def test_gompertz_hazard() -> None:
    hf = bd.GompertzHazard(alpha_const=1e-5, alpha_energy=1e-6, gamma=1.0, beta=1e-5)
    age = jnp.array([10.0, 110.0, 10.0, 110.0])
    energy = jnp.array([0.0, 0.0, 20.0, 20.0])
    hazard = hf(age, energy)
    chex.assert_trees_all_close(
        hazard,
        jnp.array(
            [
                (1e-5 + 1e-6 * jnp.exp(0.0)) * jnp.exp(1e-5 * 10.0),
                (1e-5 + 1e-6 * jnp.exp(0.0)) * jnp.exp(1e-5 * 110.0),
                (1e-5 + 1e-6 * jnp.exp(-20.0)) * jnp.exp(1e-5 * 10.0),
                (1e-5 + 1e-6 * jnp.exp(-20.0)) * jnp.exp(1e-5 * 110.0),
            ]
        ),
    )


def test_elgompertz_hazard() -> None:
    hf = bd.ELGompertzHazard(alpha=1.0, scale=1.0, e0=3.0, alpha_age=1e-6, beta=1e-5)
    age = jnp.array([10.0, 110.0, 10.0, 110.0])
    energy = jnp.array([0.0, 0.0, 20.0, 20.0])
    hazard = hf(age, energy)
    chex.assert_trees_all_close(
        hazard,
        jnp.array(
            [
                1.0 - (1.0 / (1.0 + jnp.exp(3.0))) + 1e-6 * jnp.exp(1e-5 * 10),
                1.0 - (1.0 / (1.0 + jnp.exp(3.0))) + 1e-6 * jnp.exp(1e-5 * 110),
                1.0 - (1.0 / (1.0 + jnp.exp(-17.0))) + 1e-6 * jnp.exp(1e-5 * 10),
                1.0 - (1.0 / (1.0 + jnp.exp(-17.0))) + 1e-6 * jnp.exp(1e-5 * 110),
            ]
        ),
    )


def test_energylogistic_birth() -> None:
    hf = bd.EnergyLogisticBirth(alpha=1.0, scale=0.1, e0=8.0)
    age = jnp.array([10.0, 110.0, 10.0, 110.0])
    energy = jnp.array([0.0, 0.0, 20.0, 20.0])
    hazard = hf(age, energy)
    chex.assert_trees_all_close(
        hazard,
        jnp.array(
            [
                0.1 / (1.0 + jnp.exp(8.0)),
                0.1 / (1.0 + jnp.exp(8.0)),
                0.1 / (1.0 + jnp.exp(-12.0)),
                0.1 / (1.0 + jnp.exp(-12.0)),
            ]
        ),
    )


def test_evaluate_hazard() -> None:
    hf = bd.ELGompertzHazard(alpha=1.0, scale=1.0, e0=3.0, alpha_age=1e-6, beta=1e-5)
    energy = jnp.array(
        [
            [0.0, 10.0, 20.0],
            [10.0, 10.0, 10.0],
            [20.0, 10.0, 0.0],
        ]
    )
    age_from = jnp.array([10.0, 10.0, 0.0])
    age_to = jnp.array([20.0, 20.0, 10.0])
    hazard = bd.evaluate_hazard(hf, age_from, age_to, energy)
    chex.assert_trees_all_close(
        hazard,
        jnp.array(
            [
                1.0 - (1.0 / (1.0 + jnp.exp(3.0))) + 1e-6 * jnp.exp(1e-5 * 10),
                1.0 - (1.0 / (1.0 + jnp.exp(3.0))) + 1e-6 * jnp.exp(1e-5 * 110),
                1.0 - (1.0 / (1.0 + jnp.exp(-17.0))) + 1e-6 * jnp.exp(1e-5 * 10),
                1.0 - (1.0 / (1.0 + jnp.exp(-17.0))) + 1e-6 * jnp.exp(1e-5 * 110),
            ]
        ),
    )
