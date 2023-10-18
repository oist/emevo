import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.circle_foraging import _make_space
from emevo.environments.phyjax2d import Space, StateDict
from emevo.environments.placement import place_agent
from emevo.environments.utils.locating import CircleCoordinate, InitLoc

N_MAX_AGENTS = 20
N_MAX_FOODS = 10


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def get_space_and_more() -> tuple[Space, StateDict, CircleCoordinate]:
    coordinate = CircleCoordinate((100.0, 100.0), 100.0)
    space, seg_state = _make_space(
        0.1,
        coordinate,
        n_max_agents=N_MAX_AGENTS,
        n_max_foods=N_MAX_FOODS,
    )
    stated = space.shaped.zeros_state().replace(segment=seg_state)
    return space, stated, coordinate


def test_place_agents(key) -> None:
    n = N_MAX_AGENTS // 2
    keys = jax.random.split(key, n)
    space, stated, coordinate = get_space_and_more()
    initloc_fn = InitLoc.GAUSSIAN((100.0, 100.0), (10.0, 10.0))

    for i, key in enumerate(keys):
        xy = place_agent(
            n_trial=10,
            agent_radius=6.0,
            coordinate=coordinate,
            initloc_fn=initloc_fn,
            key=key,
            shaped=space.shaped,
            stated=stated,
        )
        assert jnp.all(xy < jnp.inf)
        circle_xy = circle_xy.at[i].set(xy)
        circle = circle.replace(p=circle.p.replace(xy=circle_xy))  # type: ignore
