"""Utilities for concise representations of PyTree containers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import numpy as np

T = TypeVar("T", bound=type)


def _array_repr(array: jax.Array | np.ndarray[Any, Any]) -> str:
    shape = ",".join(str(size) for size in array.shape)
    return f"{array.dtype}[{shape}]"


def _container_name(value: Any) -> str | None:
    if dataclasses.is_dataclass(value) or hasattr(value, "_fields"):
        return f"{type(value).__name__}(...)"
    return None


def _short_repr(value: Any) -> str:
    if isinstance(value, (jax.Array, np.ndarray)):
        return _array_repr(value)

    container_name = _container_name(value)
    if container_name is not None:
        return container_name

    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, tuple):
        return f"tuple[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, Callable):
        return getattr(value, "__name__", type(value).__name__)
    return repr(value)


def _field_names(value: Any) -> tuple[str, ...]:
    if hasattr(value, "_fields"):
        return value._fields
    if dataclasses.is_dataclass(value):
        return tuple(field.name for field in dataclasses.fields(value))
    raise TypeError(f"{type(value).__name__} is not a dataclass or NamedTuple")


def pytree_repr(value: Any) -> str:
    """Return a concise, value-free representation of a PyTree container."""

    fields = ",\n  ".join(
        f"{name}={_short_repr(getattr(value, name))}" for name in _field_names(value)
    )
    return f"{type(value).__name__}(\n  {fields}\n)"


def compact_pytree_repr(cls: T) -> T:
    """Add concise ``__repr__`` and ``__str__`` methods to a PyTree class."""

    def __repr__(self: Any) -> str:
        return pytree_repr(self)

    setattr(cls, "__repr__", __repr__)
    setattr(cls, "__str__", __repr__)
    return cls
