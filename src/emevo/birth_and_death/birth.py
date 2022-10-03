from __future__ import annotations

import dataclasses
from typing import Callable, Protocol, TypeVar

from emevo.birth_and_death.statuses import HasAgeAndEnergy

# typevar for status
S = TypeVar("S", contravariant=True)


class BirthFunction(Protocol[S]):
    def __call__(self, status: S) -> float:
        """Birth function b(t)"""
        ...


@dataclasses.dataclass
class GeneralizedLogstic(BirthFunction[HasAgeAndEnergy]):
    def __call__(self, status: HasAgeAndEnergy) -> float:
        return super().__call__(status)
