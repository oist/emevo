""" Set of death functions for default statuses
"""
import typing as t

import numpy as np
import scipy.special as sc

from .core import DeathProbFn, Status


def _gompertz_makeham_cdf(
    x: float,
    *,
    alpha: float,
    beta: float,
    lambda_: float,
) -> float:
    return -sc.expm1(-lambda_ * x - (alpha / beta) * np.exp(beta * x))


def gompertz_makeham_beta_energy(
    energy_to_beta: t.Callable[[float], float] = lambda energy: energy ** 2,
    beta_max: float = 256,
    alpha: float = 0.04,
    lambda_: float = 0.01,
    scale: float = 400.0,
) -> DeathProbFn:
    """Use gompertz-makeham distribution to model the death of an agent.
    Take α and λ, and β is modelled by the energy_level ** 2.
    """

    def death_prob_fn(status: Status) -> bool:
        beta = min(beta_max, energy_to_beta(status.energy_level))
        x = float(status.age) / scale
        cd = _gompertz_makeham_cdf(x, alpha=alpha, beta=beta, lambda_=lambda_)
        return cd

    return death_prob_fn
