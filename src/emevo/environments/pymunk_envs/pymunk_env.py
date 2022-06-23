"""A common interface for pymunk envs."""

from typing import Protocol

import pymunk


class PymunkEnv(Protocol):
    def get_space(self) -> pymunk.Space:
        ...
