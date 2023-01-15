import abc
from typing import Any, Generic

from emevo.body import Body
from emevo.env import LOC
from emevo.status import Status


class Newborn(abc.ABC, Generic[LOC]):
    """A class that contains information of birth type."""

    def __init__(
        self,
        parent: Body,
        parental_status: Status | tuple[Status, Status],
        time_to_birth: int,
        info: Any = None,
    ) -> None:
        self.parent = parent
        self.parental_status = parental_status
        self.info = info
        self.time_to_birth = time_to_birth

    def is_ready(self) -> bool:
        """Return if the newborn is ready to be born or not."""
        return self.time_to_birth == 0

    @abc.abstractmethod
    def location(self) -> LOC:
        """Notify the newborn that the timestep has moved on."""
        pass

    def step(self) -> None:
        """Notify the newborn that the timestep has moved on."""
        if self.time_to_birth == 0:
            raise RuntimeError("Newborn.step is called when it's ready")
        self.time_to_birth -= 1


class Oviparous(Newborn[LOC]):
    """A newborn stays in an egg for a while and will be born."""

    def __init__(
        self,
        parent: Body,
        parental_status: Status | tuple[Status, Status],
        time_to_birth: int,
        info: Any = None,
    ) -> None:
        super().__init__(parent, parental_status, time_to_birth, info=info)
        self._location = parent.location()

    def location(self) -> LOC:
        return self._location


class Viviparous(Newborn[LOC]):
    """A newborn stays in a parent's body for a while and will be born."""

    def location(self) -> LOC:
        return self.parent.location()
