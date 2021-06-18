import typing as t

import numpy as np

from emevo import birth_and_death as bd
from emevo import Body


DEFAULT_ENERGY_LEVEL: int = 10
EMPTY_ARRAY: np.ndarray = np.array(())


class FakeBody(Body):
    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return None


def _get_manager(n_bodies: int = 5, **other_args) -> bd.Manager:
    manager = bd.Manager(
        default_status=bd.Status(energy_level=float(DEFAULT_ENERGY_LEVEL)),
        is_dead=lambda status: status.energy_level < 0.5,
        **other_args,
    )
    for _ in range(n_bodies):
        manager.register(FakeBody(name="FakeBody"))
    return manager


def _oviparous(time_to_birth: int) -> bd.Oviparous:
    return bd.Oviparous(
        EMPTY_ARRAY,
        EMPTY_ARRAY,
        time_to_birth=time_to_birth,
    )


def test_asexual() -> None:
    # 10 steps to death, 11 steps to birth, 3 steps to newborn
    STEPS_TO_DEATH: int = 10
    STEPS_TO_BIRTH: int = 3

    def repr_fn(status: bd.Status) -> t.Optional[bd.Oviparous]:
        if status.energy_level > float(DEFAULT_ENERGY_LEVEL + STEPS_TO_DEATH):
            return _oviparous(STEPS_TO_BIRTH)
        else:
            return None

    manager = _get_manager(asexual_repr_fn=repr_fn)
    bodies = list(manager.available_bodies())
    for step_idx in range(STEPS_TO_DEATH):
        for body_idx, body in enumerate(bodies):
            diff = -1.0 if body_idx % 2 == 1 else 1.0
            manager.update(body, energy_level=diff)
        for body in bodies:
            assert not manager.asexual_repr(body)
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
        manager.update(body, energy_level=+1.0)

    for body in bodies:
        assert manager.asexual_repr(body)

    for step_idx in range(STEPS_TO_BIRTH):
        _, newborns = manager.step()
        if step_idx == STEPS_TO_BIRTH - 1:
            assert len(newborns) == 3
        else:
            assert len(newborns) == 0
