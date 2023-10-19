import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.circle_foraging import _make_space
from emevo.environments.phyjax2d import Space, StateDict
from emevo.environments.placement import place_agent, place_food
from emevo.environments.utils.food_repr import ReprLoc
from emevo.environments.utils.locating import CircleCoordinate, InitLoc

N_MAX_AGENTS = 20
N_MAX_FOODS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


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
        agent_radius=AGENT_RADIUS,
        food_radius=FOOD_RADIUS,
    )
    stated = space.shaped.zeros_state().replace(segment=seg_state)
    return space, stated, coordinate


def test_place_agents(key) -> None:
    n = N_MAX_AGENTS // 2
    keys = jax.random.split(key, n)
    space, stated, coordinate = get_space_and_more()
    initloc_fn = InitLoc.UNIFORM(CircleCoordinate((100.0, 100.0), 95.0))
    assert stated.circle is not None
    for i, key in enumerate(keys):
        xy = place_agent(
            n_trial=10,
            agent_radius=AGENT_RADIUS,
            coordinate=coordinate,
            initloc_fn=initloc_fn,
            key=key,
            shaped=space.shaped,
            stated=stated,
        )
        assert jnp.all(xy < jnp.inf), stated.circle.p.xy
        stated = stated.nested_replace("circle.p.xy", stated.circle.p.xy.at[i].set(xy))

    is_active = jnp.concatenate(
        (
            jnp.ones(n, dtype=bool),
            jnp.zeros(N_MAX_AGENTS + N_MAX_FOODS - n, dtype=bool),
        )
    )
    stated = stated.nested_replace("circle.is_active", is_active)

    # test no overwrap each other
    contact_data = space.check_contacts(stated)
    assert jnp.all(contact_data.contact.penetration <= 0.0)


def test_place_foods(key) -> None:
    n = N_MAX_FOODS // 2
    keys = jax.random.split(key, n)
    space, stated, coordinate = get_space_and_more()
    reprloc_fn, reprloc_state = ReprLoc.UNIFORM(CircleCoordinate((100.0, 100.0), 95.0))
    assert stated.circle is not None
    for i, key in enumerate(keys):
        xy = place_food(
            n_trial=10,
            food_radius=FOOD_RADIUS,
            coordinate=coordinate,
            reprloc_fn=reprloc_fn,
            reprloc_state=reprloc_state,
            key=key,
            shaped=space.shaped,
            stated=stated,
        )
        assert jnp.all(xy < jnp.inf), stated.circle.p.xy
        stated = stated.nested_replace(
            "circle.p.xy",
            stated.circle.p.xy.at[i + N_MAX_AGENTS].set(xy),
        )

    is_active = jnp.concatenate(
        (
            jnp.zeros(N_MAX_AGENTS, dtype=bool),
            jnp.ones(n, dtype=bool),
            jnp.zeros(N_MAX_FOODS - n, dtype=bool),
        )
    )
    stated = stated.nested_replace("circle.is_active", is_active)

    # test no overwrap each other
    contact_data = space.check_contacts(stated)
    assert jnp.all(contact_data.contact.penetration <= 0.0)
