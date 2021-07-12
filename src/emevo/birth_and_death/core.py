"""Core components of birth_and_death, including Manager
"""

import abc
import dataclasses
import datetime as dt
import typing as t

import numpy as np

from emevo.body import Body
from emevo.environment import Encount

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
    success_prob: t.Callable[[Status], float]
    produce: t.Callable[[Status], Newborn]


@dataclasses.dataclass(frozen=True)
class SexualReprManager:
    success_prob: t.Callable[[t.Tuple[Status, Status], Encount], float]
    produce: t.Callable[[t.Tuple[Status, Status], Encount], Newborn]


@dataclasses.dataclass
class Manager:
    """
    Manager manages energy level, birth and death of agents.
    Note that Manager does not manage matings.
    """

    default_status_fn: t.Callable[[], Status]
    death_prob_fn: t.Callable[[Status], float]
    repr_manager: t.Union[AsexualReprManager, SexualReprManager]
    rng: t.Callable[[], float] = np.random.rand
    statuses: t.Dict[Body, Status] = dataclasses.field(default_factory=dict)
    pending_newborns: t.List[Newborn] = dataclasses.field(default_factory=list)

    def available_bodies(self) -> t.Iterable[Body]:
        return self.statuses.keys()

    def register(self, body: Body, status: t.Optional[Status] = None) -> None:
        if status is None:
            status = self.statuses[body] = self.default_status_fn()
        self.statuses[body] = status

    def update_status(self, body: Body, **updates) -> None:
        self.statuses[body].update(**updates)

    def reproduce(self, arg: t.Union[Body, Encount]) -> bool:
        if isinstance(arg, Encount):
            statuses = tuple((self.statuses[body] for body in arg.bodies))
            args = statuses, arg
        else:
            args = (self.statuses[arg],)
        success_prob = self.repr_manager.success_prob(*args)
        if self.rng() < success_prob:
            newborn = self.repr_manager.produce(*args)
            self.pending_newborns.append(newborn)
            return True
        else:
            return False

    def step(self) -> t.Tuple[t.List[DeadBody], t.List[Newborn]]:
        deads, newborns = [], []

        for body, status in self.statuses.items():
            status.step()
            if self.rng() < self.death_prob_fn(status):
                deads.append(DeadBody(body, status, dt.datetime.now()))

        for dead in deads:
            del self.statuses[dead.body]

        for i, newborn in enumerate(self.pending_newborns):
            newborn.step()
            if newborn.is_ready():
                newborns.append(newborn)

        for newborn in newborns:
            self.pending_newborns.remove(newborn)

        return deads, newborns
