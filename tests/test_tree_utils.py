import jax.numpy as jnp

from emevo.environments.circle_foraging import CFObs
from emevo.tree_utils import compact_pytree_repr


@compact_pytree_repr
class ExampleTuple(tuple):
    _fields = ("array", "items")

    def __new__(cls, array: object, items: object) -> "ExampleTuple":
        return tuple.__new__(cls, (array, items))

    array = property(lambda self: self[0])
    items = property(lambda self: self[1])


def test_compact_pytree_repr() -> None:
    value = ExampleTuple(jnp.zeros((2, 3)), [1, 2])

    assert repr(value) == "ExampleTuple(\n  array=float32[2,3],\n  items=list[2]\n)"
    assert str(value) == repr(value)


def test_cfobs_repr() -> None:
    obs = CFObs(
        sensor=jnp.zeros((2, 3, 4)),
        collision=jnp.zeros((2, 4, 5), dtype=bool),
        velocity=jnp.zeros((2, 2)),
        angle=jnp.zeros(2),
        angular_velocity=jnp.zeros(2),
        energy=jnp.zeros(2),
    )

    output = repr(obs)
    assert output.startswith("CFObs(\n")
    assert "sensor=float32[2,3,4]" in output
    assert "collision=bool[2,4,5]" in output
