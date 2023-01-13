from __future__ import annotations

import pytest

from emevo.status import Status


@pytest.mark.parametrize("capacity", (10, 100))
def test_status(capacity: float) -> None:
    status = Status(age=0.0, energy=0.0, capacity=capacity)
    for _ in range(200):
        status.update(energy_delta=1.0)
        assert status.energy >= 0.0 and status.energy <= capacity

    for _ in range(300):
        status.update(energy_delta=-1.0)
        assert status.energy >= 0.0 and status.energy <= capacity
