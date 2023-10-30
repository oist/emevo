import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.circle_foraging import _make_physics
from emevo.environments.phyjax2d import Space, StateDict
from emevo.environments.placement import place_agent, place_food
from emevo.environments.utils.food_repr import ReprLoc
from emevo.environments.utils.locating import CircleCoordinate, InitLoc
