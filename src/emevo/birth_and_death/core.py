"""Core components of birth_and_death, including Manager
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable

import numpy as np

from emevo.birth_and_death.newborn import Newborn
from emevo.body import Body, Encount


@dataclasses.dataclass
class Status:
    """Default status implementation with age and energy."""

    age: float
    energy: float

    def step(self) -> None:
        """Get older."""
        self.age += 1

    def share(self, ratio: float) -> float:
        """Share some portion of energy."""
        shared = self.energy * ratio
        self.update(energy_delta=-shared)
        return shared

    def update(self, *, energy_delta: float) -> None:
        """Update energy."""
        energy = self.energy + energy_delta
        self.energy = max(0.0, energy)


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
        hazard_fn: Callable[[Status], float],
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
        initial_status_fn: Callable[..., Status],
        hazard_fn: Callable[[Status], float],
        birth_fn: Callable[[Status], float],
        produce_fn: Callable[[Status, Body], Newborn],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        super().__init__(initial_status_fn, hazard_fn, rng)
        self._birth_fn = birth_fn
        self._produce_fn = produce_fn

    def _try_reproduce(self, body: Body) -> bool:
        success_prob = self._birth_fn(self._statuses[body])
        if self._rng() < success_prob:
            newborn = self._produce_fn(self._statuses[body], body)
            self._pending_newborns.append(newborn)
            return True
        else:
            return False

    def reproduce(self, body: Body | Iterable[Body]) -> list[Body]:
        """
        Try asexual reproducation from a body or an iterator over bodies.
        Return a list of bodies that reproduced themselves.
        """
        if isinstance(body, Body):
            bodies = [body]
        else:
            bodies = body
        return [body for body in bodies if self._try_reproduce(body)]


class SexualReprManager(_BaseManager):
    def __init__(
        self,
        initial_status_fn: Callable[..., Status],
        hazard_fn: Callable[[Status], float],
        birth_fn: Callable[[Status, Status], float],
        produce_fn: Callable[[Status, Status, Encount], Newborn],
        rng: Callable[[], float] = np.random.rand,
    ) -> None:
        super().__init__(initial_status_fn, hazard_fn, rng)
        self._birth_fn = birth_fn
        self._produce_fn = produce_fn

    def _try_reproduce(self, encount: Encount) -> bool:
        s_a, s_b = map(lambda body: self._statuses[body], encount)
        success_prob = self._birth_fn(s_a, s_b)
        if self._rng() < success_prob:
            newborn = self._produce_fn(s_a, s_b, encount)
            self._pending_newborns.append(newborn)
            return True
        else:
            return False

    def reproduce(self, encount: Encount | Iterable[Encount]) -> list[Encount]:
        """Try asexual reproducation from an encount or an iterator over encounts."""
        if isinstance(encount, Encount):
            encounts = [encount]
        else:
            encounts = encount
        return [encount for encount in encounts if self._try_reproduce(encount)]
