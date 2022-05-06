"""
Abstract environment API.
Two points are made in the design:
1.
2. Since EmEvo assumes that each agent does not have full access to the
environmental state, a state class should have 'observe' API that
"""
import abc

from typing import Dict, List, Optional, Tuple, Union

from emevo.body import Body, Encount
from emevo.types import Action, Array, Info, Location, Observation


class State(abc.ABC):
    """An abstraction layer for 'state' of the environment."""

    pass


class Env(abc.ABC):
    """Abstract API for emevo environments"""

    @abc.abstractmethod
    def bodies(self) -> List[Body]:
        """Returns all 'alive' bodies in the environment"""
        pass

    @abc.abstractmethod
    def step(
        self,
        state: State,
        actions: Dict[Body, Action],
    ) -> Tuple[State, List[Encount]]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def observe(self, body: Body, state: State) -> Tuple[Observation, Info]:
        """Construct the observation from the state"""
        pass

    @abc.abstractmethod
    def reset(self, seed: Optional[Union[Array, int]]) -> State:
        """Do some initialization"""
        pass

    @abc.abstractmethod
    def born(self, location: Location, state: State) -> Tuple[Body, State]:
        """Taken some locations, place newborns in the environment."""
        pass

    @abc.abstractmethod
    def dead(self, body: Body, state: State) -> State:
        """Remove a dead body from the environment."""
        pass

    @abc.abstractmethod
    def is_extinct(self) -> bool:
        """Return if agents are extinct"""
        pass

    def render(self, mode: str) -> Union[None, Array]:
        """Render something to GUI or file"""
        pass
