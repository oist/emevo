import typing

import chex
import jax
import jax.numpy as jnp
import pytest

from emevo import TimeStep, make
from emevo.environments.circle_foraging import (
    CFObs,
    CFState,
    CircleForaging,
    _observe_closest,
    _observe_closest_with_food_labels,
    get_sensor_obs,
)

N_MAX_AGENTS = 10
AGENT_RADIUS = 10
FOOD_RADIUS = 4


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(43)


def reset_env(key: chex.PRNGKey) -> tuple[CircleForaging, CFState, TimeStep[CFObs]]:
    #     x
    #   O x O
    # O x O  O  (O: agent, x: food)
    env = make(
        "CircleForaging-v0",
        env_shape="square",
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=5,
        agent_loc_fn=(
            "periodic",
            [40.0, 60.0],
            [60.0, 90.0],
            [80.0, 60.0],
            [100.0, 90.0],
            [120.0, 60.0],
        ),
        food_loc_fn=(
            "periodic",
            [60.0, 60.0],
            [80.0, 90.0],
            [80.0, 120.0],
        ),
        food_num_fn=("constant", 3),
        foodloc_interval=20,
        agent_radius=AGENT_RADIUS,
        food_radius=FOOD_RADIUS,
    )
    state, timestep = env.reset(key)
    return typing.cast(CircleForaging, env), state, timestep


def reset_multifood_env(
    key: chex.PRNGKey,
) -> tuple[CircleForaging, CFState, TimeStep[CFObs]]:
    #   O x
    # O x O x  (O: agent, x: food)
    env = make(
        "CircleForaging-v0",
        env_shape="square",
        n_max_agents=N_MAX_AGENTS,
        n_initial_agents=3,
        agent_loc_fn=(
            "periodic",
            [40.0, 60.0],
            [60.0, 90.0],
            [80.0, 60.0],
        ),
        n_food_sources=3,
        food_loc_fn=[
            ("periodic", [60.0, 60.0]),  # 0
            ("periodic", [80.0, 90.0]),  # 1
            ("periodic", [100.0, 60.0]),  # 2
        ],
        food_num_fn=[
            ("constant", 1),
            ("constant", 1),
            ("constant", 1),
        ],
        agent_radius=AGENT_RADIUS,
        food_radius=FOOD_RADIUS,
        observe_food_label=True,
    )
    state, timestep = env.reset(key)
    return typing.cast(CircleForaging, env), state, timestep


def test_observe_closest(key: chex.PRNGKey) -> None:
    env, state, _ = reset_env(key)

    def observe(p1: list[float], p2: list[float]) -> jax.Array:
        return _observe_closest(
            env._physics.shaped,
            jnp.array(p1),
            jnp.array(p2),
            state.physics,
        )

    obs = observe([40.0, 10.0], [40.0, 30.0])
    chex.assert_trees_all_close(obs, jnp.ones(3) * -1)
    obs = observe([40.0, 10.0], [40.0, 110.0])
    chex.assert_trees_all_close(obs, jnp.array([0.6, -1.0, -1.0]))
    obs = observe([60.0, 10.0], [60.0, 110.0])
    chex.assert_trees_all_close(obs, jnp.array([-1.0, 0.54, -1.0]))
    obs = observe([130.0, 60.0], [230.0, 60.0])
    chex.assert_trees_all_close(obs, jnp.array([-1.0, -1.0, 0.3]))


def test_observe_closest_with_foodlabels(key: chex.PRNGKey) -> None:
    env, state, _ = reset_multifood_env(key)

    def observe(p1: list[float], p2: list[float]) -> jax.Array:
        return _observe_closest_with_food_labels(
            3,
            env._physics.shaped,
            jnp.array(p1),
            jnp.array(p2),
            state.physics,
        )

    obs = observe([40.0, 10.0], [40.0, 110.0])
    chex.assert_trees_all_close(obs, jnp.array([0.6, -1.0, -1.0, -1.0, -1.0]))
    obs = observe([60.0, 10.0], [60.0, 110.0])
    chex.assert_trees_all_close(obs, jnp.array([-1.0, 0.54, -1.0, -1.0, -1.0]))
    obs = observe([100.0, 10.0], [100.0, 110.0])
    chex.assert_trees_all_close(obs, jnp.array([-1.0, -1.0, -1.0, 0.54, -1.0]))
    obs = observe([100.0, 90.0], [0.0, 90.0])
    chex.assert_trees_all_close(obs, jnp.array([-1.0, -1.0, 0.84, -1.0, -1.0]))


def test_sensor_obs(key: chex.PRNGKey) -> None:
    env, state, _ = reset_env(key)
    sensor_obs = get_sensor_obs(
        env._physics.shaped,
        3,
        (-90, 90),
        100.0,
        None,
        state.physics,
    )
    chex.assert_shape(sensor_obs, (30, 3))
    # Food is to the right/left of the circle
    chex.assert_trees_all_close(
        sensor_obs[0],
        sensor_obs[3],
        sensor_obs[8],
        sensor_obs[11],
        jnp.array([-1.0, 0.94, -1.0]),
    )
    # Food is above the circle
    chex.assert_trees_all_close(sensor_obs[7], jnp.array([-1.0, 0.84, -1.0]))
    # They can see each other
    chex.assert_trees_all_close(
        sensor_obs[6],
        sensor_obs[14],
        jnp.array([0.8, -1.0, -1.0]),
    )
    # Walls
    chex.assert_trees_all_close(sensor_obs[2], jnp.array([-1.0, -1.0, 0.7]))
    chex.assert_trees_all_close(sensor_obs[5], jnp.array([-1.0, -1.0, 0.5]))
    chex.assert_trees_all_close(sensor_obs[9], jnp.array([-1.0, -1.0, 0.1]))
    chex.assert_trees_all_close(
        sensor_obs[4],
        sensor_obs[10],
        jnp.array([-1.0, -1.0, 0.0]),
    )
    chex.assert_trees_all_close(sensor_obs[12], jnp.array([-1.0, -1.0, 0.3]))
    # Nothing
    chex.assert_trees_all_close(
        sensor_obs[1],
        sensor_obs[13],
        jnp.array([-1.0, -1.0, -1.0]),
    )


def test_sensor_obs_with_foodlabels(key: chex.PRNGKey) -> None:
    env, state, _ = reset_multifood_env(key)
    sensor_obs = get_sensor_obs(
        env._physics.shaped,
        3,
        (-90, 90),
        100.0,
        3,
        state.physics,
    )
    chex.assert_shape(sensor_obs, (30, 5))
    # Food 0 is to the right/left
    chex.assert_trees_all_close(
        sensor_obs[0],
        sensor_obs[8],
        jnp.array([-1.0, 0.94, -1.0, -1.0, -1.0]),
    )
    # Food 1 is to the right
    chex.assert_trees_all_close(
        sensor_obs[3],
        jnp.array([-1.0, -1.0, 0.94, -1.0, -1.0]),
    )
    # Food 1 is above
    chex.assert_trees_all_close(
        sensor_obs[7],
        jnp.array([-1.0, -1.0, 0.84, -1.0, -1.0]),
    )
    # Food 2 is to the right
    chex.assert_trees_all_close(
        sensor_obs[6],
        jnp.array([-1.0, -1.0, -1.0, 0.94, -1.0]),
    )


def test_encount_and_collision(key: chex.PRNGKey) -> None:
    #     x
    #   O x←3
    # O x 2→ ←4
    env, state, _ = reset_env(key)
    step = jax.jit(env.step)
    act1 = jnp.zeros((10, 2)).at[2, 0].set(20).at[3:5, 1].set(20)
    while True:
        state, ts = step(state, act1)
        assert jnp.all(jnp.logical_not(ts.encount))
        if state.physics.circle.p.angle[4] >= jnp.pi * 0.45:
            break

    act2 = jnp.zeros((10, 2)).at[2:5].set(20.0)
    p2p4_ok, p3_ok = False, False
    n_iter = 0
    for _ in range(100):
        p2 = state.physics.circle.p.xy[2]
        p3 = state.physics.circle.p.xy[3]
        p4 = state.physics.circle.p.xy[4]
        state, ts = step(state, act2)
        if not p2p4_ok and jnp.linalg.norm(p2 - p4) <= 2 * AGENT_RADIUS:
            assert bool(ts.encount[2, 4]), (p2, p3, p4)
            assert bool(ts.encount[4, 2]), (p2, p3, p4)
            assert bool(ts.obs.collision[2, 0, -1]), (p2, p3, p4)
            assert bool(ts.obs.collision[4, 0, 0]), (p2, p3, p4)
            p2p4_ok = True

        p3_to_food = jnp.linalg.norm(p3 - jnp.array([80.0, 90.0]))
        if not p3_ok and p3_to_food <= AGENT_RADIUS + FOOD_RADIUS:
            assert bool(ts.obs.collision[3, 1, 0]), (p2, p3, p4)
            p3_ok = True

        if p2p4_ok and p3_ok:
            break
        n_iter += 1

    assert n_iter < 99


def test_collision_with_foodlabels(key: chex.PRNGKey) -> None:
    #   O->x
    # O->x O->x
    env, state, _ = reset_multifood_env(key)
    step = jax.jit(env.step)
    # Rotate agent to right
    act1 = jnp.zeros((10, 2)).at[:3, 0].set(20)
    while True:
        state, ts = step(state, act1)
        assert jnp.all(jnp.logical_not(ts.encount))
        if state.physics.circle.p.angle[0] <= jnp.pi * 1.51:
            break

    # Move it to right
    act2 = act1.at[:3, 1].set(20.0)
    n_iter = 0
    for _ in range(100):
        p = state.physics.circle.p.xy[:3]
        state, ts = step(state, act2)

        to_food = jnp.linalg.norm(
            p - jnp.array([[60.0, 60.0], [80.0, 90.0], [100.0, 60.0]]),
            axis=1,
        )
        if jnp.any(ts.obs.collision):
            assert jnp.all(to_food <= AGENT_RADIUS + FOOD_RADIUS + 0.1)
            chex.assert_trees_all_close(
                ts.obs.collision[:3, :, -1],
                jnp.array(
                    [
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                    ]
                ),
            )
            break

        n_iter += 1

    assert n_iter < 99


def test_asarray(key: chex.PRNGKey) -> None:
    env, _, timestep = reset_env(key)
    obs = timestep.obs.as_array()
    obs_shape = env.obs_space.flatten().shape[0]
    chex.assert_shape(obs, (N_MAX_AGENTS, obs_shape))
