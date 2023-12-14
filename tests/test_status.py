import jax.numpy as jnp
import pytest

from emevo.env import init_status


@pytest.mark.parametrize(
    "n, capacity",
    [(1, 10.0), (1, 100.0), (10, 10.0), (10, 100.0)],
)
def test_status_clipping(n: int, capacity: float) -> None:
    status = init_status(n=n, max_n=n, init_energy=0.0)
    for _ in range(200):
        status.update(energy_delta=jnp.ones(n), capacity=capacity)
        assert jnp.all(status.energy >= 0.0)
        assert jnp.all(status.energy <= capacity)

    for _ in range(300):
        status.update(energy_delta=jnp.ones(n) * -1.0, capacity=capacity)
        assert jnp.all(status.energy >= 0.0)
        assert jnp.all(status.energy <= capacity)
