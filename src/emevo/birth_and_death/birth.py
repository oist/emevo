from __future__ import annotations

import dataclasses
from typing import Callable, Protocol, TypeVar

from emevo.birth_and_death.statuses import HasEnergy

# typevar for status
S = TypeVar("S", contravariant=True)


class BirthFunction(Protocol[S]):
    def __call__(self, status: S) -> float:
        """Birth function b(t)"""
        ...
