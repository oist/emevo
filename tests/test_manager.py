from __future__ import annotations

from functools import partial
from typing import Callable

import pytest

from emevo import Encount
from emevo import birth_and_death as bd
from emevo._test_utils import FakeBody

DEFAULT_ENERGY_LEVEL: int = 10


@pytest.fixture
def status_fn():
    return partial(bd.Status, age=1, energy=DEFAULT_ENERGY_LEVEL)


@pytest.fixture
def hazard_fn() -> bd.death.HazardFunction:
    return bd.death.Deterministic(0.5, 100.0)


def _add_bodies(manager, n_bodies: int = 5) -> None:
    for _ in range(n_bodies):
        manager.register(FakeBody())


def test_asexual(
    status_fn: Callable[[], bd.Status],
    hazard_fn: Callable[[bd.Status], float],
) -> None:
    """Test the most basic setting: Asexual reproduction + Oviparous birth"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = DEFAULT_ENERGY_LEVEL
    STEPS_TO_BIRTH: int = 3

    manager = bd.AsexualReprManager(
        initial_status_fn=status_fn,
        hazard_fn=hazard_fn,
        birth_fn=lambda status: float(
            status.energy > DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH
        ),
        produce_fn=lambda _status, body: bd.Oviparous(
            parent=body,
            time_to_birth=STEPS_TO_BIRTH,
        ),
    )
    _add_bodies(manager)

    bodies = list(manager.available_bodies())
    for step_idx in range(STEPS_TO_DEATH):
        for body_idx, body in enumerate(bodies):
            manager.update_status(
                body,
                energy_delta=-1.0 if body_idx % 2 == 1 else 1.0,
            )
        parents = manager.reproduce(bodies)
        assert len(parents) == 0
        deads, newborns = manager.step()
        if step_idx == STEPS_TO_DEATH - 1:
            assert len(deads) == 2
            assert len(newborns) == 0
            for dead in deads:
                assert dead.body not in manager._statuses
                bodies.remove(dead.body)
        else:
            assert len(deads) == 0, f"{step_idx}"
            assert len(newborns) == 0

    for body in bodies:
        manager.update_status(body, energy_delta=1.0)

    parents = manager.reproduce(bodies)
    for body in bodies:
        assert body in parents

    for step_idx in range(STEPS_TO_BIRTH):
        _, newborns = manager.step()
        if step_idx == STEPS_TO_BIRTH - 1:
            assert len(newborns) == 3
        else:
            assert len(newborns) == 0


@pytest.mark.parametrize("newborn_cls", [bd.Oviparous, bd.Viviparous])
def test_sexual(
    status_fn: Callable[[], bd.Status],
    hazard_fn: Callable[[bd.Status], float],
    newborn_cls: type[bd.Newborn],
) -> None:
    """Test Sexual reproduction"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = 10
    STEPS_TO_BIRTH: int = 3

    def success_prob(
        status_a: bd.Status,
        status_b: bd.Status,
    ) -> float:
        threshold = float(DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH)
        if status_a.energy > threshold and status_b.energy > threshold:
            return 1.0
        else:
            return 0.0

    def produce(_sa, _sb, encount: Encount) -> bd.Newborn:
        return newborn_cls(
            parent=encount.a,
            time_to_birth=STEPS_TO_BIRTH,
        )

    manager = bd.SexualReprManager(
        initial_status_fn=status_fn,
        hazard_fn=hazard_fn,
        birth_fn=success_prob,
        produce_fn=produce,
    )

    _add_bodies(manager)

    bodies = list(manager.available_bodies())
    for step_idx in range(STEPS_TO_DEATH):
        for body_idx, body in enumerate(bodies):
            diff = -1.0 if body_idx % 2 == 1 else 1.0
            manager.update_status(body, energy_delta=diff)
        deads, newborns = manager.step()
        if step_idx == STEPS_TO_DEATH - 1:
            assert len(deads) == 2
            assert len(newborns) == 0
            for dead in deads:
                assert dead.body not in manager._statuses
                bodies.remove(dead.body)
        else:
            assert len(deads) == 0
            assert len(newborns) == 0

    for body in bodies:
        manager.update_status(body, energy_delta=1.0)

    assert len(manager.reproduce(Encount(bodies[0], bodies[1]))) == 1

    for step_idx in range(STEPS_TO_BIRTH):
        _, newborns = manager.step()
        if step_idx == STEPS_TO_BIRTH - 1:
            assert len(newborns) == 1
        else:
            assert len(newborns) == 0
