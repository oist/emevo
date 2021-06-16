"""
This module provides Admin, a utility class to manage birth and death of agents.
"""

import abc
import dataclasses
import typing as t

import numpy as np


@dataclasses.dataclass()
class Admin:
    """
    Admin is a utility for managing birth and death.
    This is an optional API and not mandatory.
    """

    pending_children: t.List[Child] = dataclasses.field(default_factory=list)

    def create_new_agent(self, gene: np.ndarray) -> Agent:
        """TODO: more information to pas"""
        pass


class Child(abc.ABC):
    """A class contains information of birth type."""

    gene: np.ndarray

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

    gene: np.ndarray
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

    gene: np.ndarray
    parent: Agent

    def is_ready(self) -> bool:
        return self.parent.is_dead


@dataclasses.dataclass()
class Viviparous(Child):
    """A child stays in a parent's body for a while and will be born."""

    gene: np.ndarray
    parent: Agent
    time_to_birth: int

    def is_ready(self) -> bool:
        return self.time_to_birth == 0 or self.parent.is_dead

    def step(self) -> None:
        if self.time_to_birth == 0:
            raise RuntimeError("Child.step is called when it's ready")
        self.time_to_birth -= 1
