from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from numpy.typing import NDArray

from emevo import Body, Encount
from emevo import birth_and_death as bd
from emevo import spaces

DEFAULT_ENERGY_LEVEL: int = 10


class FakeBody(Body):
    def __init__(self, name: str) -> None:
        act_space = spaces.BoxSpace(
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )
        obs_space = spaces.BoxSpace(
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )
        super().__init__(act_space, obs_space, name)

    def location(self) -> NDArray:
        return np.array(())


@pytest.fixture
def status_fn():
    return partial(bd.statuses.AgeAndEnergy, age=1, energy=DEFAULT_ENERGY_LEVEL)


@pytest.fixture
def death_prob_fn():
    return bd.death_functions.hunger_or_infirmity(0.5, 100.0)


def _add_bodies(manager, n_bodies: int = 5) -> None:
    for _ in range(n_bodies):
        manager.register(FakeBody(name="FakeBody"))


def test_asexual(status_fn, death_prob_fn) -> None:
    """Test the most basic setting: Asexual reproduction + Oviparous birth"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = DEFAULT_ENERGY_LEVEL
    STEPS_TO_BIRTH: int = 3

    manager = bd.AsexualReprManager(
        initial_status_fn=status_fn,
        death_prob_fn=death_prob_fn,
        success_prob=lambda status: float(
            status.energy > DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH  # type: ignore
        ),
        produce=lambda _status: bd.Oviparous(context=(), time_to_birth=STEPS_TO_BIRTH),
    )
    _add_bodies(manager)

    bodies = list(manager.available_bodies())
    for step_idx in range(STEPS_TO_DEATH):
        for body_idx, body in enumerate(bodies):
            manager.update_status(
                body,
                energy_update=-1.0 if body_idx % 2 == 1 else 1.0,
            )
        for body in bodies:
            assert not manager.reproduce(body)
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
        manager.update_status(body, energy_update=1.0)

    for body in bodies:
        assert manager.reproduce(body)

    for step_idx in range(STEPS_TO_BIRTH):
        _, newborns = manager.step()
        if step_idx == STEPS_TO_BIRTH - 1:
            assert len(newborns) == 3
        else:
            assert len(newborns) == 0


@pytest.mark.parametrize("newborn_kind", ["oviparous", "viviparous"])
def test_sexual(status_fn, death_prob_fn, newborn_kind: str) -> None:
    """Test Sexual reproduction"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = 10
    STEPS_TO_BIRTH: int = 3

    def success_prob(
        statuses: tuple[bd.Status, bd.Status],
        _encount: Encount,
    ) -> float:
        threshold = float(DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH)
        energy_ok = all(map(lambda status: status.energy > threshold, statuses))  # type: ignore
        return 1.0 if energy_ok else 0.0

    if newborn_kind == "oviparous":

        def produce(*_args) -> bd.Oviparous:
            return bd.Oviparous(context=(), time_to_birth=STEPS_TO_BIRTH)

    elif newborn_kind == "viviparous":

        def produce(_, encount) -> bd.Viviparous:
            return bd.Viviparous(
                context=(),
                parent=encount.a,
                time_to_birth=STEPS_TO_BIRTH,
            )

    else:

        raise ValueError(f"Unknown newborn kind {newborn_kind}")

    manager = bd.SexualReprManager(
        initial_status_fn=status_fn,
        death_prob_fn=death_prob_fn,
        success_prob=success_prob,
        produce=produce,
    )
    _add_bodies(manager)

    bodies = list(manager.available_bodies())
    for step_idx in range(STEPS_TO_DEATH):
        for body_idx, body in enumerate(bodies):
            diff = -1.0 if body_idx % 2 == 1 else 1.0
            manager.update_status(body, energy_update=diff)
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
        manager.update_status(body, energy_update=1.0)

    assert manager.reproduce(Encount(bodies[0], bodies[1]))

    for step_idx in range(STEPS_TO_BIRTH):
        _, newborns = manager.step()
        if step_idx == STEPS_TO_BIRTH - 1:
            assert len(newborns) == 1
        else:
            assert len(newborns) == 0
