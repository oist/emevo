from __future__ import annotations

import dataclasses
from typing import Protocol

from emevo.birth_and_death.statuses import Status


class BirthFunction(Protocol):
    def __call__(self, status: Status) -> float:
        """Birth function b(t)"""
        ...


@dataclasses.dataclass
class GeneralizedLogstic(BirthFunction):
    def __call__(self, status: Status) -> float:
        return super().__call__(status)
