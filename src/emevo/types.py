from collections.abc import Sequence
from typing import Any, Protocol

import jax

DType = jax.numpy.dtype


class SupportsDType(Protocol):
    @property
    def dtype(self) -> DType:
        ...


DTypeLike = DType | SupportsDType
PyTree = Any
Self = Any
Index = int | jax.Array | Sequence[int]
