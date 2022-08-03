"""Core components of birth_and_death, including Manager
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable

import numpy as np

from emevo.birth_and_death.newborn import Newborn
from emevo.birth_and_death.statuses import Status
from emevo.body import Body, Encount


@dataclasses.dataclass(frozen=True)
class DeadBody:
    """Dead Body"""

    body: Body
    status: Status


class _BaseManager:
    """
    Manager manages energy level, birth and death of agents.
    Note that Manager does not manage matings.
    """

    def __init__(
        self,
        initial_status_fn: Callable[..., Status],
        death_prob_fn: Callable[[Status], float],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        self._initial_status_fn = initial_status_fn
        self._death_prob_fn = death_prob_fn
        self._rng = rng
        self._statuses = {}
        self._pending_newborns = []

    def available_bodies(self) -> Iterable[Body]:
        return self._statuses.keys()

    def register(self, body: Body, *args, **kwargs) -> None:
        self._statuses[body] = self._initial_status_fn(*args, **kwargs)

    def step(self) -> tuple[list[DeadBody], list[Newborn]]:
        deads, newborns = [], []

        for body, status in self._statuses.items():
            status.step()
            if self._rng() < self._death_prob_fn(status):
                deads.append(DeadBody(body, status))

        for dead in deads:
            del self._statuses[dead.body]

        for newborn in self._pending_newborns:
            newborn.step()
            if newborn.is_ready():
                newborns.append(newborn)

        for newborn in newborns:
            self._pending_newborns.remove(newborn)

        return deads, newborns

    def update_status(self, body: Body, **updates) -> None:
        self._statuses[body].update(**updates)


class AsexualReprManager(_BaseManager):
    def __init__(
        self,
        initial_status_fn: Callable[..., Status],
        death_prob_fn: Callable[[Status], float],
        success_prob: Callable[[Status], float],
        produce: Callable[[Status], Newborn],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        super().__init__(initial_status_fn, death_prob_fn, rng)
        self._success_prob = success_prob
        self._produce = produce

    def reproduce(self, body: Body) -> Newborn | None:
        success_prob = self._success_prob(self._statuses[body])
        if self._rng() < success_prob:
            newborn = self._produce(self._statuses[body])
            self._pending_newborns.append(newborn)
            return newborn
        else:
            return None


class SexualReprManager(_BaseManager):
    def __init__(
        self,
        initial_status_fn: Callable[..., Status],
        death_prob_fn: Callable[[Status], float],
        success_prob: Callable[[tuple[Status, Status], Encount], float],
        produce: Callable[[tuple[Status, Status], Encount], Newborn],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        super().__init__(initial_status_fn, death_prob_fn, rng)
        self._success_prob = success_prob
        self._produce = produce

    def reproduce(self, encount: Encount) -> Newborn | None:
        statuses = tuple((self._statuses[body] for body in encount))
        success_prob = self._success_prob(statuses, encount)
        if self._rng() < success_prob:
            newborn = self._produce(statuses, encount)
            self._pending_newborns.append(newborn)
            return newborn
        else:
            return None
