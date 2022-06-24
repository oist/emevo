"""Core components of birth_and_death, including Manager
"""
from __future__ import annotations

import abc
import dataclasses
import datetime as dt
from typing import Callable, Iterable

import numpy as np

from emevo.birth_and_death.newborn import Newborn
from emevo.body import Body, Encount


class Status(abc.ABC):
    def step(self) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass


@dataclasses.dataclass(frozen=True)
class DeadBody:
    """R.I.P."""

    body: Body
    status: Status
    dead_time: dt.datetime


@dataclasses.dataclass(frozen=True)
class AsexualReprManager:
    success_prob: Callable[[Status, Body], float]
    produce: Callable[[Status, Body], Newborn]


@dataclasses.dataclass(frozen=True)
class SexualReprManager:
    success_prob: Callable[[tuple[Status, Status], Encount], float]
    produce: Callable[[tuple[Status, Status], Encount], Newborn]


@dataclasses.dataclass
class Manager:
    """
    Manager manages energy level, birth and death of agents.
    Note that Manager does not manage matings.
    """

    status_fn: Callable[..., Status]
    death_prob_fn: Callable[[Status], float]
    repr_manager: AsexualReprManager | SexualReprManager
    rng: Callable[[], float] = np.random.rand
    statuses: dict[Body, Status] = dataclasses.field(default_factory=dict)
    pending_newborns: list[Newborn] = dataclasses.field(default_factory=list)

    def available_bodies(self) -> Iterable[Body]:
        return self.statuses.keys()

    @property
    def is_asexual(self) -> bool:
        return isinstance(self.repr_manager, AsexualReprManager)

    def register(self, body: Body, *status_args, **status_kwargs) -> None:
        self.statuses[body] = self.status_fn(*status_args, **status_kwargs)

    def reproduce(self, body_or_encount: Body | Encount) -> Newborn | None:
        if isinstance(body_or_encount, Encount):
            statuses = tuple((self.statuses[body] for body in body_or_encount))
            args = statuses, body_or_encount
        else:
            assert isinstance(
                body_or_encount, Body
            ), f"invalid type as body_or_encount: {type(body_or_encount)}"
            args = self.statuses[body_or_encount], body_or_encount
        success_prob = self.repr_manager.success_prob(*args)
        if self.rng() < success_prob:
            newborn = self.repr_manager.produce(*args)
            self.pending_newborns.append(newborn)
            return newborn
        else:
            return None

    def step(self) -> tuple[list[DeadBody], list[Newborn]]:
        deads, newborns = [], []

        for body, status in self.statuses.items():
            status.step()
            if self.rng() < self.death_prob_fn(status):
                deads.append(DeadBody(body, status, dt.datetime.now()))

        for dead in deads:
            del self.statuses[dead.body]

        for newborn in self.pending_newborns:
            newborn.step()
            if newborn.is_ready():
                newborns.append(newborn)

        for newborn in newborns:
            self.pending_newborns.remove(newborn)

        return deads, newborns

    def update_status(self, body: Body, **updates) -> None:
        self.statuses[body].update(**updates)

    def stats(self, stats_fn: Callable[[Status], float]) -> dict[str, float]:
        stats = np.array(list(map(stats_fn, self.statuses.values())))
        return {"Average": np.mean(stats), "Max": np.max(stats), "Min": np.min(stats)}
