import enum
import typing as t

import numpy as np

from emevo.body import Body

ReproduceFn = t.Callable[
    [t.List[Body], t.Callable[[np.ndarray], bool], np.random.RandomState],
    t.List[np.ndarray],
]


class ReprKind(str, enum.Enum):
    logistic: str = "logistic"
    constrained: str = "constrained"


def logistic_repr(growth_rate: float, capacity: float) -> ReproduceFn:
    def reproduce_fn(
        archeas: t.List[Body],
        overlapped_with: t.Callable[[np.ndarray], bool],
        np_random: np.random.RandomState,
    ) -> t.List[np.ndarray]:
        n_archea = len(archeas)
        dn_dt = growth_rate * n_archea * (1 - n_archea / capacity)
        res = []
        for _ in range(max(0, int(dn_dt))):
            position = np_random.uniform(size=2)
            while overlapped_with(position):
                position = np_random.uniform(size=2)
            res.append(position)
        return res

    return reproduce_fn


def bacteria_constrained_repr(
    initial_n_bacterias: int,
    bacteria_growth_rate: float,
    bacteria_capacity: float,
) -> ReproduceFn:
    @dataclasses.dataclass
    class BacteriaState:
        num: float
        growth_rate: t.ClassVar[float] = bacteria_growth_rate
        capacity: t.ClassVar[float] = bacteria_capacity
        delta_min: t.ClassVar[float] = 0.01

        def growth(self) -> None:
            delta = self.growth_rate * self.num * (1 - self.num / self.capacity)
            self.num += max(delta, self.delta_min)

    bacteria = BacteriaState(float(initial_n_bacterias))

    def reproduce_fn(
        archeas: t.List[Body],
        overlapped_with: t.Callable[[np.ndarray], bool],
        np_random: np.random.RandomState,
    ) -> t.List[np.ndarray]:
        n_archea = len(archeas)
        res = []
        for idx in np_random.permutation(n_archea):
            prob = min(int(bacteria.num), bacteria.capacity) / (bacteria.capacity + 1)
            if np_random.rand() <= prob:
                x, y = archeas[idx].position
                low = np.maximum([0.0, 0.0], [x - 0.1, y - 0.1])
                high = np.minimum([1.0, 1.0], [x + 0.1, y + 0.1])
                position = np_random.uniform(low=low, high=high)
                while overlapped_with(position):
                    position = np_random.uniform(low=low, high=high)
                res.append(position)
                bacteria.num -= 1.0

        # It archeas are extinct, give it a chance
        if n_archea == 0:
            prob = min(int(bacteria.num), bacteria.capacity) / (bacteria.capacity + 1)
            if np_random.rand() <= prob:
                position = np_random.uniform(size=2)
                while overlapped_with(position):
                    position = np_random.uniform(size=2)
                res.append(position)
                bacteria.num -= 1.0

        bacteria.growth()
        return res

    return reproduce_fn
