import typing as t
from functools import partial

import numpy as np
import pytest

from gym import spaces

from emevo import Body, Encount, birth_and_death as bd

DEFAULT_ENERGY_LEVEL: int = 10


class FakeBody(Body):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))

    @property
    def position(self) -> np.ndarray:
        return np.array(())


def _get_manager(repr_manager, n_bodies: int = 5) -> bd.Manager:
    manager = bd.Manager(
        default_status_fn=partial(
            bd.statuses.AgeAndEnergy,
            age=1,
            energy=DEFAULT_ENERGY_LEVEL,
        ),
        death_prob_fn=bd.death_functions.hunger_or_infirmity(0.5, 100.0),
        repr_manager=repr_manager,
    )
    for _ in range(n_bodies):
        manager.register(FakeBody(name="FakeBody"))
    return manager


def test_asexual() -> None:
    """Test the most basic setting: Asexual reproduction + Oviparous birth"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = DEFAULT_ENERGY_LEVEL
    STEPS_TO_BIRTH: int = 3

    manager = _get_manager(
        bd.AsexualReprManager(
            success_prob=lambda status: float(
                status.energy > DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH
            ),
            produce=lambda _: bd.Oviparous(context=(), time_to_birth=STEPS_TO_BIRTH),
        )
    )
    bodies = list(manager.available_bodies())
    for step_idx in range(STEPS_TO_DEATH):
        for body_idx, body in enumerate(bodies):
            manager.update_status(
                body, energy_update=-1.0 if body_idx % 2 == 1 else 1.0
            )
        for body in bodies:
            assert not manager.reproduce(body)
        deads, newborns = manager.step()
        if step_idx == STEPS_TO_DEATH - 1:
            assert len(deads) == 2
            assert len(newborns) == 0
            for dead in deads:
                assert dead.body not in manager.statuses
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
def test_sexual(newborn_kind: callable) -> None:
    """Test Sexual reproduction"""

    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = 10
    STEPS_TO_BIRTH: int = 3

    def success_prob(
        statuses: t.Tuple[bd.Status, bd.Status],
        encount: Encount,
    ) -> float:
        threshold = float(DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH)
        energy_ok = all(map(lambda status: status.energy > threshold, statuses))
        distance_ok = encount.distance < 1.0
        return 1.0 if energy_ok and distance_ok else 0.0

    if newborn_kind == "oviparous":

        repr_manager = bd.SexualReprManager(
            success_prob=success_prob,
            produce=lambda _, __: bd.Oviparous(
                context=(), time_to_birth=STEPS_TO_BIRTH
            ),
        )

    elif newborn_kind == "viviparous":

        repr_manager = bd.SexualReprManager(
            success_prob=success_prob,
            produce=lambda _, encount: bd.Viviparous(
                context=(), parent=encount.bodies[0], time_to_birth=STEPS_TO_BIRTH
            ),
        )

    else:

        raise ValueError(f"Unknown newborn kind {newborn_kind}")

    manager = _get_manager(repr_manager=repr_manager)
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
                assert dead.body not in manager.statuses
                bodies.remove(dead.body)
        else:
            assert len(deads) == 0, f"{i}"
            assert len(newborns) == 0

    for body in bodies:
        manager.update_status(body, energy_update=1.0)

    assert manager.reproduce(Encount((bodies[0], bodies[1]), 0.5))

    for step_idx in range(STEPS_TO_BIRTH):
        _, newborns = manager.step()
        if step_idx == STEPS_TO_BIRTH - 1:
            assert len(newborns) == 1
        else:
            assert len(newborns) == 0
