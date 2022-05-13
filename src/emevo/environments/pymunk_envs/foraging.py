import dataclasses

from typing import Dict, List, Optional, Tuple, Union

import pymunk

from numpy.typing import NDArray

from emevo.body import Body, Encount
from emevo.env import Env, Observation
from emevo.environments.utils import FoodReprFn, ReprMethods
from emevo.types import Info, Location


class FgBody(Body):
    def __init__(self) -> None:
        pass


@dataclasses.dataclass
class FgObs(Observation):
    sensor: NDArray

    def flatten(self) -> NDArray:
        return self.sensor


class Foraging(Env[NDArray, FgBody, FgObs]):
    def __init__(
        self,
        n_initial_bodies: int = 6,
        food_repr_fn: FoodReprFn = ReprMethods.constant(10),
    ) -> None:
        self._food_repr_fn = food_repr_fn
        self._space = pymunk.Space()
        self._bodies = []

    def bodies(self) -> List[FgBody]:
        return self._bodies

    def step(self, actions: Dict[Body, NDArray]) -> Tuple[List[Encount], Info]:
        pass

    def observe(self, body: Body) -> Tuple[FgObs, Info]:
        pass

    def reset(self, seed: Optional[Union[NDArray, int]] = None) -> None:
        pass

    def born(self, location: Location) -> Tuple[FgBody, FgObs]:
        pass

    def dead(self, body: FgBody) -> None:
        pass

    def is_extinct(self) -> bool:
        pass
