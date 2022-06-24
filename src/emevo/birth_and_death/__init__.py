"""Utilities for handling birth and death of agents.
"""

from . import birth_functions, death_functions, statuses  # noqa
from .core import (  # noqa
    AsexualReprManager,
    DeadBody,
    Manager,
    SexualReprManager,
    Status,
)
from .newborn import Newborn, Oviparous, Viviparous  # noqa
