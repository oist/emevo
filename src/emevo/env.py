"""
Abstract environment API.
"""
import abc

from typing import Any, Dict, Generic, List, Optional, TypeVar

from numpy.typing import ArrayLike

from emevo.body import Body, Encount

Self = Any
BODY = TypeVar("BODY", bound=Body)
LOC = TypeVar("LOC")


class Env(abc.ABC, Generic[BODY, LOC]):
    """Abstract API for emevo environments"""

    def __init__(self, *args, **kwargs) -> None:
        # To supress PyRight errors in registry
        pass

    @abc.abstractmethod
    def bodies(self) -> List[BODY]:
        """Returns all 'alive' bodies in the environment"""
        pass

    @abc.abstractmethod
    def step(self, actions: Dict[BODY, ArrayLike]) -> List[Encount]:
        """
        Step the simulator by 1-step, taking the state and actions from each body.
        Returns the next state and all encounts.
        """
        pass

    @abc.abstractmethod
    def observe(self, body: BODY) -> ArrayLike:
        """Construct the observation from the state"""
        pass

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Do some initialization"""
        pass

    @abc.abstractmethod
    def born(self, location: LOC, generation: int) -> Optional[BODY]:
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

    @abc.abstractmethod
    def visualizer(self, *args, **kwargs) -> "Visualizer":
        """Create a visualizer for the environment"""
        pass


class Visualizer:
    def close(self) -> None:
        """Close this visualizer"""
        pass

    def render(self, env: Any) -> Any:
        """Render image"""
        raise NotImplementedError("render is not implemented")

    def show(self) -> None:
        """Open a GUI window"""
        pass
