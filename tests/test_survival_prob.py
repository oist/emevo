import numpy as np
import pytest

from emevo import birth_and_death as bd
from emevo.birth_and_death.population import cumulative_hazard
from emevo.status import Status

THRESHOLD: float = 1e-4


@pytest.mark.parametrize(
    "hazard_fn",
    [
        bd.death.SimplifiedGompertz(),
        bd.death.SeparatedGompertz(),
        bd.death.Constant(),
        bd.death.EnergyLogistic(),
        bd.death.Gompertz(),
    ],
)
def test_survival_prob(hazard_fn: bd.death.HazardFunction) -> None:
    for age in [100, 1000, 10000]:
        for energy in [0.0, 5.0, 10.0]:
            status = Status(age=age, energy=energy)
            analytical_solution = hazard_fn.survival(status)
            numerical_cum_h = cumulative_hazard(hazard_fn, energy=energy, max_age=age)
            numerical_solution = np.exp(-numerical_cum_h)
            print(hazard_fn.cumulative(status), numerical_cum_h)
            assert (
                abs(analytical_solution - numerical_solution) < THRESHOLD
            ), f"Age: {age} Energy: {energy}"
