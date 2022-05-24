"""Core components of birth_and_death, including Manager
"""

import abc
import dataclasses
import datetime as dt

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from emevo.body import Body, Encount

from .newborn import Newborn


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
    success_prob: Callable[[Tuple[Status, Status], Encount], float]
    produce: Callable[[Tuple[Status, Status], Encount], Newborn]


@dataclasses.dataclass
class Manager:
    """
    Manager manages energy level, birth and death of agents.
    Note that Manager does not manage matings.
    """

    status_fn: Callable[..., Status]
    death_prob_fn: Callable[[Status], float]
    repr_manager: Union[AsexualReprManager, SexualReprManager]
    rng: Callable[[], float] = np.random.rand
    statuses: Dict[Body, Status] = dataclasses.field(default_factory=dict)
    pending_newborns: List[Newborn] = dataclasses.field(default_factory=list)

    def available_bodies(self) -> Iterable[Body]:
        return self.statuses.keys()

    @property
    def is_asexual(self) -> bool:
        return isinstance(self.repr_manager, AsexualReprManager)

    def register(self, body: Body, *status_args, **status_kwargs) -> None:
        self.statuses[body] = self.status_fn(*status_args, **status_kwargs)

    def reproduce(self, body_or_encount: Union[Body, Encount]) -> Optional[Newborn]:
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

    def step(self) -> Tuple[List[DeadBody], List[Newborn]]:
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

    def stats(self, stats_fn: Callable[[Status], float]) -> Dict[str, float]:
        stats = np.array(list(map(stats_fn, self.statuses.values())))
        return {"Average": np.mean(stats), "Max": np.max(stats), "Min": np.min(stats)}
