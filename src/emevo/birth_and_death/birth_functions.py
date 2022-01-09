import typing as t

import numpy as np

from emevo import Body, Encount

from .statuses import AgeAndEnergy


def _scaled_log(value: float, scale: float) -> float:
    return np.log(1 + max(value, 0.0) * scale)


def log_prod(
    scale_energy: float,
    scale_prob: float,
) -> t.Callable[[t.Tuple[AgeAndEnergy, AgeAndEnergy], Encount], float]:
    def success_prob(
        statuses: t.Tuple[AgeAndEnergy, AgeAndEnergy],
        encount: Encount,
    ) -> float:
        log_e1, log_e2 = map(
            lambda status: _scaled_log(status.energy, scale_energy), statuses
        )
        return min(1.0, log_e1 * log_e2 * scale_prob)

    return success_prob


def log(scale_energy: float, scale_prob: float) -> t.Callable[[AgeAndEnergy], float]:
    def success_prob(status: AgeAndEnergy, _body: Body) -> float:
        return min(1.0, _scaled_log(status.energy, scale_energy) * scale_prob)

    return success_prob
