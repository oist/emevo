import dataclasses

from typing import Dict, List, Optional, Tuple, Union

import pymunk

from numpy.typing import NDArray

from emevo.body import Body, Encount
from emevo.env import Env, Observation
from emevo.types import Info, Location
from emevo.environments.utils import FoodReprFn


class ParticleBody(Body):
    def __init__(self) -> None:
        pass


@dataclasses.dataclass
class ParticleObs(Observation):
    sensor: NDArray

    def flatten(self) -> NDArray:
        return self.sensor


class ParticleForaging(Env[NDArray, ParticleBody, ParticleObs]):
    def __init__(
        self,
        n_initial_bodies: int = 6,
        food_repr: 
    ) -> None:
        self._space = pymunk.Space()
        self._bodies = []

    def bodies(self) -> List[ParticleBody]:
        return self._bodies

    def step(self, actions: Dict[Body, NDArray]) -> Tuple[List[Encount], Info]:
        pass

    def observe(self, body: Body) -> Tuple[ParticleObs, Info]:
        pass

    def reset(self, seed: Optional[Union[NDArray, int]] = None) -> None:
        pass

    def born(self, location: Location) -> Tuple[ParticleBody, ParticleObs]:
        pass

    def dead(self, body: ParticleBody) -> None:
        pass

    def is_extinct(self) -> bool:
        pass
