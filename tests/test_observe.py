import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.circle_foraging import _make_physics
from emevo.environments.phyjax2d import Space, StateDict
