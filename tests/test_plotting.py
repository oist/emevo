import matplotlib.pyplot as plt
import numpy as np
import pytest

from emevo import birth_and_death as bd
from emevo.plotting import vis_hazard_2d


@pytest.mark.parametrize("method", ["hazard", "cumulative hazard", "survival"])
def test_vis_hazard_2d(method: str) -> None:
    hazard_fn = bd.GompertzHazard(alpha=1e-5, beta=1e-3)
    fig, ax = plt.subplots()
    line, text = vis_hazard_2d(
        ax,
        hazard_fn,
        age_max=100,
        hazard_max=0.02,
        n_discr=11,
        method=method,
        shown_params={"alpha": 1e-5},
    )

    age = line.get_xdata()
    cumulative = (1e-5 / 1e-3) * (np.exp(1e-3 * age) - 1.0)
    expected = {
        "hazard": 1e-5 * np.exp(1e-3 * age),
        "cumulative hazard": cumulative,
        "survival": np.exp(-cumulative),
    }[method]
    np.testing.assert_allclose(age, np.linspace(0.0, 100.0, 11))
    np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-6, atol=1e-8)
    assert text is not None
    assert text.get_text() == "alpha: 1.00e-05"
    if method == "survival":
        assert ax.get_ylim() == (0.0, 1.0)
    else:
        assert ax.get_ylim() == (0.0, 0.02)
    plt.close(fig)


def test_vis_hazard_2d_update() -> None:
    hazard_fn = bd.GompertzHazard(alpha=1e-5)
    fig, ax = plt.subplots()
    line, text = vis_hazard_2d(ax, hazard_fn, initial=False)

    assert line.axes is ax
    assert text is None
    assert not ax.xaxis.get_gridlines()[0].get_visible()
    plt.close(fig)


def test_vis_hazard_2d_rejects_unknown_method() -> None:
    hazard_fn = bd.GompertzHazard(alpha=1e-5)
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Unsupported method unknown"):
        vis_hazard_2d(ax, hazard_fn, method="unknown")
    plt.close(fig)
