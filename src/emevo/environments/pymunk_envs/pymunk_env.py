import abc

from typing import Dict, List, Optional, Tuple, Union

import pymunk

from emevo.body import Body, Encount
from emevo.env import Env
from emevo.types import Action, Array, Info, Location, Observation


class PymunkBody(Body):
    def __init__(self) -> None:
        pass


class PymunkEnv(Env, abc.ABC):
    def __init__(self) -> None:
        self._space = pymunk.Space()
        self._bodies: List[PymunkBody] = []

    def bodies(self) -> List[Body]:
        return self._bodies

    def step(self, actions: Dict[Body, Action]) -> List[Encount]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    def observe(self, body: Body) -> Tuple[Observation, Info]:
        """Construct the observation from the state"""
        pass

    def reset(self, seed: Optional[Union[Array, int]] = None) -> State:
        """Do some initialization"""
        pass

    def born(self, location: Location) -> Tuple[Body]:
        """Taken some locations, place newborns in the environment."""
        pass

    def dead(self, body: Body) -> None:
        """Remove a dead body from the environment."""
        pass

    def is_extinct(self) -> bool:
        """Return if agents are extinct"""
        pass
