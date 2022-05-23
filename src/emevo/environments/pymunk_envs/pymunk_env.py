"""A common interface for pymunk envs."""
import abc

import pymunk


class PymunkEnv(abc.ABC):
    @abc.abstractmethod
    def get_space(self) -> pymunk.Space:
        pass
