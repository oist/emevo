from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from loguru import logger
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
from pymunk.vec2d import Vec2d

from emevo import _test_utils as utils
from emevo.environments import CircleForaging


def almost_equal(actual: NDArray, desired: float) -> NDArray:
    diff = np.abs(actual - desired)
    return diff < 1e-6


def assert_either_a_or_b(
    actual: NDArray,
    desired_a: float,
    desired_b: float,
    require_both: bool = True,
) -> None:
    almost_equal_to_a = almost_equal(actual, desired_a)
    almost_equal_to_b = almost_equal(actual, desired_b)
    almost_equal_to_a_or_b = np.logical_or(almost_equal_to_a, almost_equal_to_b)
    if not np.all(almost_equal_to_a_or_b):
        raise AssertionError(
            f"Some elements of {actual} are not equal to"
            + f" either of {desired_a} or {desired_b}"
        )
    if require_both:
        if not np.any(almost_equal_to_a):
            raise AssertionError(
                f"At least one element of {actual} should be {desired_a}"
            )
        if not np.any(almost_equal_to_b):
            raise AssertionError(
                f"At least one element of {actual} should be {desired_b}"
            )


logger.enable("emevo")

# Not to depend on the default argument
AGENT_RADIUS: float = 8.0
DT: float = 0.05
FOOD_RADIUS: float = 4.0
MAX_ABS_VELOCITY: float = 1.0
N_SENSORS: int = 4
SENSOR_LENGTH: float = 10.0
YLIM: tuple[float, float] = 0.0, 200


@pytest.fixture
def env() -> CircleForaging:
    return utils.predefined_env(
        agent_radius=AGENT_RADIUS,
        sensor_length=SENSOR_LENGTH,
        n_agent_sensors=N_SENSORS,
        n_physics_steps=1,
        max_abs_velocity=MAX_ABS_VELOCITY,
        ylim=YLIM,
        dt=DT,
    )


@dataclasses.dataclass
class DebugLogHandler:
    logs: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        logger.add(self, format="{message}", level="DEBUG")

    def __call__(self, message: str) -> None:
        self.logs.append(message)

    def __contains__(self, query: str) -> bool:
        for log in self.logs:
            if query in log:
                return True
        return False

    def once(self, query: str) -> bool:
        n_occurance = 0
        for log in self.logs:
            if query in log:
                n_occurance += 1
        return n_occurance == 1


def test_birth(env: CircleForaging) -> None:
    """
    Test that
    1. we can't place body if it overlaps another object
    2. we can place body otherwise.
    """
    assert len(env.bodies()) == 3
    body = next(filter(lambda body: body.info().position.x > 100.0, env.bodies()))
    place = body.info().position
    assert env.born(place, 1) is None
    assert env.born(place + Vec2d(6.5, -5.0), 1) is None
    assert env.born(place + Vec2d(-16.0, 0.0), 1) is not None
    env.dead(body)
    assert env.born(place, 1) is not None


def test_death(env: CircleForaging) -> None:
    """
    A  F

    A  A
    Kill the lower right body.
    """
    assert len(env.bodies()) == 3
    body = next(filter(lambda body: body.info().position.x > 100.0, env.bodies()))
    env.dead(body)
    assert len(env.bodies()) == 2


def test_eating(env: CircleForaging) -> None:
    """
    Confirm that eating detection (collision of an agent to a food) correctly works.
    Initially, food (F) and agent (A) are placed like
    A  F

    A  A
    , and we add force only to the lower right agent.
    """
    handler = DebugLogHandler()
    body = next(filter(lambda body: body.info().position.x > 100.0, env.bodies()))
    food = next(iter(env._foods.keys()))
    actions = {body: np.array([0.0, 1.0])}
    already_ate = False

    while True:
        _ = env.step(actions)
        observation = env.observe(body)

        if already_ate:
            if "created" in handler:
                break
        else:
            if "eaten" in handler:
                assert_almost_equal(observation.collision[1], 1.0)
                already_ate = True
            else:
                assert observation.collision[1] == 0.0

        if already_ate:
            continue

        distance_to_food = (
            food.position.y - body.info().position.y - AGENT_RADIUS - FOOD_RADIUS
        )
        if SENSOR_LENGTH < distance_to_food:
            assert_almost_equal(observation.sensor[1], 0.0)
        else:
            alpha = max(0.0, distance_to_food / SENSOR_LENGTH)
            assert_either_a_or_b(observation.sensor[1], 1.0 - alpha, 0.0)

    assert handler.once("eaten")
    assert handler.once("created")


def test_encounts(env: CircleForaging) -> None:
    """
    Confirm that encount detection (collision between agents) correctly works.
    Again, food (F) and agent (A) are placed like
    A  F

    A  A
    , and we add forces to two agents on the left, so that they collide.
    """
    body_higher, body_lower = None, None
    for body in env.bodies():
        pos = body.info().position
        if pos.x < 100 and 100 < pos.y:
            body_higher = body
        elif pos.x < 100 and pos.y < 100:
            body_lower = body

    assert body_higher is not None and body_lower is not None
    actions = {body_higher: np.array([0.0, -1.0]), body_lower: np.array([0.0, 1.0])}

    while True:
        encounts = env.step(actions)
        obs_high = env.observe(body_higher)
        obs_low = env.observe(body_lower)
        # Test touch sensor
        distance = body_higher.info().position.y - body_lower.info().position.y

        # Test encount
        if len(encounts) > 0:
            assert_almost_equal(obs_low.collision[0], 1.0)
            assert_almost_equal(obs_high.collision[0], 1.0)
            assert len(encounts) == 1
            a, b = encounts[0]
            assert (body_higher is a and body_lower is b) or (
                body_lower is a and body_higher is b
            )
            break
        else:
            assert_almost_equal(obs_low.collision[0], 0.0)
            assert_almost_equal(obs_high.collision[0], 0.0)

        if SENSOR_LENGTH + AGENT_RADIUS * 2 < distance:
            assert_almost_equal(obs_low.sensor[0], 0.0)
            assert_almost_equal(obs_high.sensor[0], 0.0)
        else:
            alpha = max(0.0, (distance - AGENT_RADIUS * 2) / SENSOR_LENGTH)
            assert_either_a_or_b(obs_low.sensor[0], 1.0 - alpha, 0.0)
            assert_either_a_or_b(obs_high.sensor[0], 1.0 - alpha, 0.0)


def test_static(env: CircleForaging) -> None:
    """
    Confirm that collision detection to walls correctly works.
    Again, food (F) and agent (A) are placed like
    A  F

    A  A
    , and we push the lower right agent to right wall.
    """
    body = next(filter(lambda body: body.info().position.x > 100.0, env.bodies()))
    actions = {body: np.array([0.0, -1.0])}

    while True:
        _ = env.step(actions)
        observation = env.observe(body)
        distance_to_wall = body.info().position.y - AGENT_RADIUS - env._WALL_RADIUS

        if SENSOR_LENGTH < distance_to_wall:
            assert_almost_equal(observation.sensor[2], 0.0)
        else:
            alpha = max(0.0, distance_to_wall / SENSOR_LENGTH)
            assert_either_a_or_b(observation.sensor[2], 1.0 - alpha, 0.0)

        # Collision
        if distance_to_wall < MAX_ABS_VELOCITY * DT:
            if np.abs(1.0 - observation.collision[2]) < 1e-6:
                break
        else:
            assert_almost_equal(observation.collision[2], 0.0)


def test_observe(env: CircleForaging) -> None:
    """
    Test the observation shape
    """
    body = env.bodies()[0]

    _ = env.step({body: np.array([0.0, -1.0])})
    observation = env.observe(body)
    assert np.asarray(observation).shape == (3 * N_SENSORS + 3 + 2 + 1 + 1,)


def test_can_place(env: CircleForaging) -> None:
    """Test that invalid position is correctly rejected"""
    assert not env._can_place(Vec2d(-10.0, -10.0), 1.0)
    assert not env._can_place(Vec2d(0.0, 0.0), 1.0)
    assert not env._can_place(Vec2d(0.9, 0.9), 1.0)
    assert env._can_place(Vec2d(1.5, 1.5), 1.0)
    assert not env._can_place(Vec2d(200.0, 200.0), 1.0)
    assert not env._can_place(Vec2d(220.0, 220.0), 1.0)
    assert not env._can_place(Vec2d(198.6, 198.6), 1.0)
    assert env._can_place(Vec2d(198.5, 198.5), 1.0)
    assert not env._can_place(Vec2d(50.0, 48.0), 5.0)
    assert env._can_place(Vec2d(50.0, 48.0), 4.0)
