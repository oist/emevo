"""A common interface for pymunk envs."""
from __future__ import annotations

from typing import Protocol

import pymunk

from emevo.environments.utils.locating import Coordinate


class PymunkEnv(Protocol):
    def get_space(self) -> pymunk.Space:
        ...

    def get_coordinate(self) -> Coordinate:
        ...

    def get_body_index(self, body: pymunk.Body) -> int | None:
        ...
