"""
birth_and_death module provides some utilities for handling birth and death of agents.
"""

from . import death_functions, repr_functions  # noqa
from .core import (  # noqa
    AsexualReprManager,
    DeadBody,
    Manager,
    SexualReprManager,
    Status,
)
from .newborn import Newborn, Oviparous, Viviparous  # noqa
from .statuses import AgeAndEnergy  # noqa
