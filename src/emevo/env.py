"""
Abstract environment API.
"""
import abc

from typing import Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from numpy.typing import ArrayLike, NDArray

from emevo.body import Body, Encount
from emevo.types import Info, Location, Shape


class Observation(abc.ABC):
    """
    Observation type.
    It should be able to return an array representation of the observation.
    """

    @abc.abstractmethod
    def as_array(self, source: Optional[ArrayLike] = None) -> ArrayLike:
        pass


ACT = TypeVar("ACT")
BODY = TypeVar("BODY", bound=Body)
OBS = TypeVar("OBS", bound=Observation)


class Env(abc.ABC, Generic[ACT, BODY, OBS]):
    """Abstract API for emevo environments"""

    def __init__(self, *args, **kwargs) -> None:
        # To supress PyRight errors in registry
        pass

    @abc.abstractmethod
    def bodies(self) -> List[BODY]:
        """Returns all 'alive' bodies in the environment"""
        pass

    @abc.abstractmethod
    def step(self, actions: Dict[BODY, ACT]) -> Tuple[List[Encount], Info]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def observe(self, body: BODY) -> Tuple[OBS, Info]:
        """Construct the observation from the state"""
        pass

    @abc.abstractmethod
    def reset(self, seed: Optional[Union[NDArray, int]] = None) -> None:
        """Do some initialization"""
        pass

    @abc.abstractmethod
    def born(self, location: Location, generation: int) -> Tuple[BODY, OBS]:
        """Taken a location, generate and place a newborn in the environment."""
        pass

    @abc.abstractmethod
    def dead(self, body: BODY) -> None:
        """Remove a dead body from the environment."""
        pass

    @abc.abstractmethod
    def is_extinct(self) -> bool:
        """Return if agents are extinct"""
        pass
