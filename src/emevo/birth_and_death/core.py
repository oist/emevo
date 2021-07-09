"""Core components of birth_and_death, including Manager
"""

import copy
import dataclasses
import datetime as dt
import typing as t

import numpy as np

from emevo.body import Body
from emevo.environment import Encount

from .newborn import Newborn


@dataclasses.dataclass
class Status:
    """
    Default implementation of agent's status.
    You can use arbitary class instead of this.
    """

    age: int
    energy_level: float
    DELTA_E: t.ClassVar[float] = -0.1

    def update(self, *, energy_update: float) -> None:
        self.energy_level += energy_update

    def step(self) -> None:
        self.age += 1
        self.energy_level += self.DELTA_E


@dataclasses.dataclass(frozen=True)
class DeadBody:
    """R.I.P."""

    body: Body
    status: Status
    dead_time: dt.datetime


DeathProbFn = t.Callable[[Status], float]


@dataclasses.dataclass(frozen=True)
class AsexualReprManager:
    success_prob: t.Callable[[Status], bool]
    produce: t.Callable[[Status], Newborn]


@dataclasses.dataclass(frozen=True)
class SexualReprManager:
    success_prob: t.Callable[[t.Tuple[Status, Status], Encount], bool]
    produce: t.Callable[[t.Tuple[Status, Status], Encount], Newborn]


@dataclasses.dataclass
class Manager:
    """
    Manager manages energy level, birth and death of agents.
    Note that Manager does not manage matings.
    """

    default_status: Status
    death_prob_fn: DeathProbFn
    repr_manager: t.Union[AsexualReprManager, SexualReprManager]
    rng: t.Callable[[], float] = np.random.rand
    statuses: t.Dict[Body, Status] = dataclasses.field(default_factory=dict)
    pending_newborns: t.List[Newborn] = dataclasses.field(default_factory=list)

    def available_bodies(self) -> t.Iterable[Body]:
        return self.statuses.keys()

    def register(self, body: Body, status: t.Optional[Status] = None) -> None:
        if status is None:
            status = self.statuses[body] = copy.deepcopy(self.default_status)
        self.statuses[body] = status

    def update_status(self, body: Body, **updates) -> None:
        self.statuses[body].update(**updates)

    def reproduction(self, arg: t.Union[Status, Encount]) -> bool:
        if isinstance(arg, Encount):
            statuses = tuple((self.statuses[body] for body in arg.bodies))
            args = statuses, arg
        else:
            args = (arg,)
        success_prob = self.repr_manager.success_prob(*args)
        if self.rng() < success_prob:
            newborn = self.repr_manager.produce(*args)
            self.pending_newborns.append(newborn)
            return True
        else:
            return False

    def step(self) -> t.Tuple[t.List[DeadBody], t.List[Newborn]]:
        deads, newborn_indices = [], []

        for body, status in self.statuses.items():
            status.step()
            death_prob = self.death_prob_fn(status)
            if self.rng() < death_prob and False:
                deads.append(DeadBody(body, status, dt.datetime.now()))

        for dead in deads:
            del self.statuses[dead.body]

        for i, newborn in enumerate(self.pending_newborns):
            newborn.step()
            if newborn.is_ready():
                newborn_indices.append(i)

        newborns = [self.pending_newborns.pop(idx) for idx in newborn_indices]

        return deads, newborns
