"""
This module provides Admin, a utility class to manage birth and death of agents.
In addition, this module provides some 'Child' API that can be used
to represent various types of birth.
"""

import dataclasses
import typing as t

from emevo.body import Body
from emevo.child import Child
from emevo.environment import Encount


IsDeadFn = t.Callable[[Body], bool]
AsexualReprFn = t.Callable[[Body], t.Optional[Child]]
SexualReprFn = t.Callable[[Encount], t.Optional[Child]]


@dataclasses.dataclass()
class Admin:
    """
    Admin manages birth and death.
    This is an optional API and not mandatory.
    """

    is_dead: IsDeadFn
    asexual_repr_fn: t.Optional[AsexualReprFn] = None
    sexual_repr_fn: t.Optional[SexualReprFn] = None
    pending_children: t.List[Child] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.asexual_repr is None and self.sexual_repr is None:
            raise ValueError("Either of asexual/sexual repr function should be set")

    def _repr_impl(self, fn: t.Optional[callable], arg: t.Any) -> bool:
        if fn is None:
            return False
        child = fn(arg)
        if child is None:
            return False
        else:
            self.pending_children.append(child)
            return True

    def asexual_repr(self, body: Body) -> bool:
        return self._repr_impl(self.asexual_repr_fn, body)

    def sexual_repr(self, encount: Encount) -> bool:
        return self._repr_impl(self.sexual_repr_fn, encount)

    def step(self) -> t.List[Child]:
        res = []
        for child in self.pending_children:
            child.step()
            if child.is_ready():
                res.append(child)
        return res
