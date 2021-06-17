"""
This module provides some utilities for handling birth and death of agents.
"""

import abc
import dataclasses
import typing as t

import numpy as np

from emevo.body import Body
from emevo.environment import Encount
from emevo.types import Gene


@dataclasses.dataclass()
class Child(abc.ABC):
    """A class that contains information of birth type."""

    gene: Gene

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return if the child is ready to be born or not."""
        pass

    def step(self) -> None:
        """Notify the child that the timestep has moved on."""
        pass


@dataclasses.dataclass()
class Oviparous(Child):
    """A child stays in an egg for a while and will be born."""

    position: np.ndarray
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass()
class Virtual(Child):
    """Virtually replace a parent's mind, reusing the body."""

    gene: Gene
    parent: Body

    def is_ready(self) -> bool:
        return self.parent.is_dead


@dataclasses.dataclass()
class Viviparous(Child):
    """A child stays in a parent's body for a while and will be born."""

    gene: Gene
    parent: Body
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0 or self.parent.is_dead

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1


@dataclasses.dataclass()
class Status:
    energy_level: float


IsDeadFn = t.Callable[[Body], bool]
AsexualReprFn = t.Callable[[Body], t.Optional[Child]]
SexualReprFn = t.Callable[[Encount], t.Optional[Child]]


@dataclasses.dataclass()
class Manager:
    """
    Manager manages energy level, birth and death of agents.
    This is an optional API and not mandatory.
    """

    is_dead: IsDeadFn
    statuses: t.Dict[Body, Status]
    asexual_repr_fn: t.Optional[AsexualReprFn] = None
    sexual_repr_fn: t.Optional[SexualReprFn] = None
    pending_children: t.List[Child] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.asexual_repr is None and self.sexual_repr is None:
            raise ValueError("Either of asexual/sexual repr function should be set")

    def _repr_impl(self, fn: t.Optional[callable], arg: t.Any) -> bool:
        if fn is None:
            return False
        child = fn(arg)
        if child is None:
            return False
        else:
            self.pending_children.append(child)
            return True

    def asexual_repr(self, body: Body) -> bool:
        return self._repr_impl(self.asexual_repr_fn, body)

    def sexual_repr(self, encount: Encount) -> bool:
        return self._repr_impl(self.sexual_repr_fn, encount)

    def step(self) -> t.List[Child]:
        res = []
        for child in self.pending_children:
            child.step()
            if child.is_ready():
                res.append(child)
        return res
