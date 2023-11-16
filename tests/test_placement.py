import chex
import jax
import jax.numpy as jnp
import pytest

from emevo.environments.circle_foraging import _make_physics
from emevo.environments.env_utils import CircleCoordinate, Locating, place
from emevo.environments.phyjax2d import Space, StateDict

N_MAX_AGENTS = 20
N_MAX_FOODS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def get_space_and_more() -> tuple[Space, StateDict, CircleCoordinate]:
    coordinate = CircleCoordinate((100.0, 100.0), 100.0)
    space, seg_state = _make_physics(
        0.1,
        coordinate,
        linear_damping=0.9,
        angular_damping=0.9,
        n_velocity_iter=4,
        n_posiiton_iter=2,
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
    initloc_fn, initloc_state = Locating.UNIFORM(CircleCoordinate((100.0, 100.0), 95.0))
    assert stated.circle is not None
    for i, key in enumerate(keys):
        xy, ok = place(
            n_trial=10,
            radius=AGENT_RADIUS,
            coordinate=coordinate,
            loc_fn=initloc_fn,
            loc_state=initloc_state,
            key=key,
            shaped=space.shaped,
            stated=stated,
        )
        assert ok, stated.circle.p.xy
        stated = stated.nested_replace("circle.p.xy", stated.circle.p.xy.at[i].set(xy))

    is_active = jnp.concatenate(
        (
            jnp.ones(n, dtype=bool),
            jnp.zeros(N_MAX_AGENTS - n, dtype=bool),
        )
    )
    stated = stated.nested_replace("circle.is_active", is_active)

    # test no overwrap each other
    contact = space.check_contacts(stated)
    assert jnp.all(contact.penetration <= 0.0)


def test_place_foods(key) -> None:
    n = N_MAX_FOODS // 2
    keys = jax.random.split(key, n)
    space, stated, coordinate = get_space_and_more()
    reprloc_fn, reprloc_state = Locating.UNIFORM(CircleCoordinate((100.0, 100.0), 95.0))
    assert stated.static_circle is not None
    for i, key in enumerate(keys):
        xy, ok = place(
            n_trial=10,
            radius=FOOD_RADIUS,
            coordinate=coordinate,
            loc_fn=reprloc_fn,
            loc_state=reprloc_state,
            key=key,
            shaped=space.shaped,
            stated=stated,
        )
        assert ok, stated.circle.p.xy
        stated = stated.nested_replace(
            "static_circle.p.xy",
            stated.static_circle.p.xy.at[i].set(xy),
        )

    stated = stated.nested_replace(
        "circle.is_active",
        jnp.zeros(N_MAX_AGENTS, dtype=bool),
    )
    is_active = jnp.concatenate(
        (
            jnp.ones(n, dtype=bool),
            jnp.zeros(N_MAX_FOODS - n, dtype=bool),
        )
    )
    stated = stated.nested_replace("static_circle.is_active", is_active)

    # test no overwrap each other
    contact = space.check_contacts(stated)
    assert jnp.all(contact.penetration <= 0.0)
