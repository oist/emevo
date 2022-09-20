"""Core components of birth_and_death, including Manager
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Generic, Iterable, TypeVar

import numpy as np

from emevo.birth_and_death.newborn import Newborn
from emevo.birth_and_death.statuses import Status
from emevo.body import Body, Encount


@dataclasses.dataclass(frozen=True)
class DeadBody:
    """Dead Body"""

    body: Body
    status: Status


STATUS = TypeVar("STATUS", bound=Status)


class _BaseManager(Generic[STATUS]):
    """
    Manager manages energy level, birth and death of agents.
    Note that Manager does not manage matings.
    """

    def __init__(
        self,
        initial_status_fn: Callable[..., STATUS],
        hazard_fn: Callable[[STATUS], float],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        self._initial_status_fn = initial_status_fn
        self._hazard_fn = hazard_fn
        self._rng = rng
        self._statuses = {}
        self._pending_newborns = []

    def available_bodies(self) -> Iterable[Body]:
        return self._statuses.keys()

    def register(self, body: Body | Iterable[Body], *args, **kwargs) -> None:
        if isinstance(body, Body):
            self._statuses[body] = self._initial_status_fn(*args, **kwargs)
        else:
            for body_i in body:
                self._statuses[body_i] = self._initial_status_fn(*args, **kwargs)

    def step(self) -> tuple[list[DeadBody], list[Newborn]]:
        deads, newborns = [], []

        for body, status in self._statuses.items():
            status.step()
            if self._rng() < self._hazard_fn(status):
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
        initial_status_fn: Callable[..., STATUS],
        hazard_fn: Callable[[STATUS], float],
        birth_fn: Callable[[STATUS], float],
        produce_fn: Callable[[STATUS, Body], Newborn],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        super().__init__(initial_status_fn, hazard_fn, rng)
        self._birth_fn = birth_fn
        self._produce_fn = produce_fn

    def _reproduce_impl(self, body: Body) -> Newborn | None:
        success_prob = self._birth_fn(self._statuses[body])
        if self._rng() < success_prob:
            newborn = self._produce_fn(self._statuses[body], body)
            self._pending_newborns.append(newborn)
            return newborn
        else:
            return None

    def reproduce(self, body: Body | Iterable[Body]) -> list[Newborn]:
        """Try asexual reproducation from a body or an iterator over bodies."""
        if isinstance(body, Body):
            bodies = [body]
        else:
            bodies = body
        res = []
        for body in bodies:
            newborn = self._reproduce_impl(body)
            if newborn is not None:
                res.append(newborn)
        return res


class SexualReprManager(_BaseManager):
    def __init__(
        self,
        initial_status_fn: Callable[..., STATUS],
        hazard_fn: Callable[[STATUS], float],
        birth_fn: Callable[[STATUS, STATUS], float],
        produce_fn: Callable[[STATUS, STATUS, Encount], Newborn],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        super().__init__(initial_status_fn, hazard_fn, rng)
        self._birth_fn = birth_fn
        self._produce_fn = produce_fn

    def _reproduce_impl(self, encount: Encount) -> Newborn | None:
        s_a, s_b = map(lambda body: self._statuses[body], encount)
        success_prob = self._birth_fn(s_a, s_b)
        if self._rng() < success_prob:
            newborn = self._produce_fn(s_a, s_b, encount)
            self._pending_newborns.append(newborn)
            return newborn
        else:
            return None

    def reproduce(self, encount: Encount | Iterable[Encount]) -> list[Newborn]:
        """Try asexual reproducation from an encount or an iterator over encounts."""
        if isinstance(encount, Encount):
            encounts = [encount]
        else:
            encounts = encount
        res = []
        for encount in encounts:
            newborn = self._reproduce_impl(encount)
            if newborn is not None:
                res.append(newborn)
        return res
