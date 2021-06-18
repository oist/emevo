"""
This module provides some utilities for handling birth and death of agents.
"""

import abc
import copy
import dataclasses
import datetime as dt
import typing as t

from emevo.body import Body
from emevo.environment import Encount


@dataclasses.dataclass()
class Newborn(abc.ABC):
    """A class that contains information of birth type."""

    context: t.Any

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return if the newborn is ready to be born or not."""
        pass

    def step(self) -> None:
        """Notify the newborn that the timestep has moved on."""
        pass


@dataclasses.dataclass()
class Oviparous(Newborn):
    """A newborn stays in an egg for a while and will be born."""

    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Newborn.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass()
class Viviparous(Newborn):
    """A newborn stays in a parent's body for a while and will be born."""

    parent: Body
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Newborn.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass()
class Status:
    """Default implementation of agent's status"""

    energy_level: float


@dataclasses.dataclass(frozen=True)
class EncountStatus:
    statuses: t.Tuple[Status, Status]
    encount: Encount


@dataclasses.dataclass(frozen=True)
class DeadBody:
    """R.I.P."""

    body: Body
    status: Status
    dead_time: dt.datetime


IsDeadFn = t.Callable[[Status], bool]
AsexualReprFn = t.Callable[[Status], t.Optional[Newborn]]
SexualReprFn = t.Callable[[t.Tuple[Status, Status], Encount], t.Optional[Newborn]]


@dataclasses.dataclass()
class Manager:
    """
    Manager manages energy level, birth and death of agents.
    This is an optional API and not mandatory.
    """

    default_status: Status
    is_dead: IsDeadFn
    statuses: t.Dict[Body, Status] = dataclasses.field(default_factory=dict)
    asexual_repr_fn: t.Optional[AsexualReprFn] = None
    sexual_repr_fn: t.Optional[SexualReprFn] = None
    pending_newborns: t.List[Newborn] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.asexual_repr is None and self.sexual_repr is None:
            raise ValueError("Either of asexual/sexual repr function should be set")

    def available_bodies(self) -> t.Iterable[Body]:
        return self.statuses.keys()

    def asexual_repr(self, body: Body) -> bool:
        return self._repr_impl(self.asexual_repr_fn, (self.statuses[body],))

    def register(self, body: Body, status: t.Optional[Status] = None) -> None:
        if status is None:
            status = self.statuses[body] = copy.deepcopy(self.default_status)
        self.statuses[body] = status

    def update(self, body: Body, **updates) -> None:
        for name, value in updates.items():
            self.statuses[body].__dict__[name] += value

    def sexual_repr(self, encount: Encount) -> bool:
        statuses = tuple((self.statuses[body] for body in encount.bodies))
        return self._repr_impl(self.sexual_repr_fn, (statuses, encount))

    def step(self) -> t.Tuple[t.List[DeadBody], t.List[Newborn]]:
        deads, newborns = [], []

        for body, status in self.statuses.items():
            if self.is_dead(status):
                deads.append(DeadBody(body, status, dt.datetime.now()))

        for dead in deads:
            del self.statuses[dead.body]

        for newborn in self.pending_newborns:
            newborn.step()
            if newborn.is_ready():
                newborns.append(newborn)
        return deads, newborns

    def _repr_impl(self, fn: t.Optional[callable], args: tuple) -> bool:
        if fn is None:
            return False
        newborn = fn(*args)
        if newborn is None:
            return False
        else:
            self.pending_newborns.append(newborn)
            return True
