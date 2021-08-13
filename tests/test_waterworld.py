import itertools

import pytest

import emevo


@pytest.mark.parametrize(
    "n_pursuers, n_evaders, n_poison, speed_features",
    list(itertools.product([0, 4, 8], [0, 4, 8], [0, 4, 8], [False, True])),
)
@pytest.mark.filterwarnings("ignore:step is called after pursuers are distinct!")
def test_waterworld(
    n_pursuers: int,
    n_evaders: int,
    n_poison: int,
    speed_features: bool,
) -> None:
    environment = emevo.make(
        "Waterworld-v0",
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        n_poison=n_poison,
        speed_features=speed_features,
    )
    if speed_features:
        correct_shape = (242,)
    else:
        correct_shape = (152,)

    # Initialize agents
    bodies = []
    for body in environment.available_bodies():
        bodies.append(body)
        obs, _ = environment.observe(body)
        assert obs.shape == correct_shape

    # Enviromental loop
    for _ in range(10):
        # Act
        for body in bodies:
            action = body.action_space.sample()
            environment.act(body, action)

        # Step
        _ = environment.step()

        # Observe
        for body in bodies:
            obs, info = environment.observe(body)
            assert obs.shape == correct_shape
