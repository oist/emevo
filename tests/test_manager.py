from __future__ import annotations

import dataclasses
from functools import partial
from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from emevo import Body, Encount
from emevo import birth_and_death as bd
from emevo import spaces

DEFAULT_ENERGY_LEVEL: int = 10


@dataclasses.dataclass
class FakeContext:
    generation: int
    location: int


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
    return bd.death.hunger_or_infirmity(0.5, 100.0)


def _add_bodies(manager, n_bodies: int = 5) -> None:
    for _ in range(n_bodies):
        manager.register(FakeBody(name="FakeBody"))


def test_asexual(
    status_fn: Callable[[], bd.statuses.AgeAndEnergy],
    death_prob_fn: Callable[[bd.statuses.AgeAndEnergy], float],
) -> None:
    """Test the most basic setting: Asexual reproduction + Oviparous birth"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = DEFAULT_ENERGY_LEVEL
    STEPS_TO_BIRTH: int = 3

    manager = bd.AsexualReprManager(
        initial_status_fn=status_fn,
        death_prob_fn=death_prob_fn,
        success_prob_fn=lambda status: float(
            status.energy > DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH
        ),
        produce_fn=lambda _status, body: bd.Oviparous(
            context=FakeContext(body.generation + 1, 0),
            time_to_birth=STEPS_TO_BIRTH,
        ),
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
def test_sexual(
    status_fn: Callable[[], bd.statuses.AgeAndEnergy],
    death_prob_fn: Callable[[bd.statuses.AgeAndEnergy], float],
    newborn_kind: str,
) -> None:
    """Test Sexual reproduction"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = 10
    STEPS_TO_BIRTH: int = 3

    def success_prob(
        status_a: bd.statuses.AgeAndEnergy,
        status_b: bd.statuses.AgeAndEnergy,
    ) -> float:
        threshold = float(DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH)
        if status_a.energy > threshold and status_b.energy > threshold:
            return 1.0
        else:
            return 0.0

    if newborn_kind == "oviparous":

        def produce_oviparous(_sa, _sb, encount: Encount) -> bd.Oviparous:
            return bd.Oviparous(
                context=FakeContext(encount.a.generation + 1, 0),
                time_to_birth=STEPS_TO_BIRTH,
            )

        manager = bd.SexualReprManager(
            initial_status_fn=status_fn,
            death_prob_fn=death_prob_fn,
            success_prob_fn=success_prob,
            produce_fn=produce_oviparous,
        )

    elif newborn_kind == "viviparous":

        def produce_viviparous(_sa, _sb, encount: Encount) -> bd.Viviparous:
            return bd.Viviparous(
                context=FakeContext(encount.a.generation + 1, 0),
                parent=encount.a,
                time_to_birth=STEPS_TO_BIRTH,
            )

        manager = bd.SexualReprManager(
            initial_status_fn=status_fn,
            death_prob_fn=death_prob_fn,
            success_prob_fn=success_prob,
            produce_fn=produce_viviparous,
        )
    else:

        raise ValueError(f"Unknown newborn kind {newborn_kind}")

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
