import typing as t

import numpy as np

from emevo.environment import Encount
from .statuses import AgeAndEnergy


def logprod_success_prob(
    scale_energy: float,
    scale_prob: float,
) -> t.Callable[[t.Tuple[AgeAndEnergy, AgeAndEnergy], Encount], float]:
    def success_prob(
        statuses: t.Tuple[AgeAndEnergy, AgeAndEnergy],
        encount: Encount,
    ) -> float:
        energy1, energy2 = map(lambda status: max(0.0, status.energy), statuses)
        scaled_energy1, scaled_energy2 = energy1 * scale_energy, energy2 * scale_energy
        log_prod_sum = np.log(1 + scaled_energy1) * np.log(1 + scaled_energy2)
        return min(1.0, log_prod_sum * scale_prob)

    return success_prob
