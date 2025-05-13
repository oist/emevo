from typing import Literal, cast

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from emevo import birth_and_death as bd

mpl.use("Agg")


class CBarRenderer:
    """Render colorbar to numpy array"""

    def __init__(
        self,
        width: float,
        height: float,
        dpi: int = 100,
    ) -> None:
        self._fig: Figure = cast(
            Figure,
            plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi),
        )
        self._ax: Axes = self._fig.add_axes([0.0, 0.2, 1.0, 0.6])  # type: ignore

    def render(self, norm: Normalize, cm: Colormap, title: str = "Value") -> None:
        """Render cbar, but don't update figure"""
        mappable = ScalarMappable(norm=norm, cmap=cm)
        self._fig.colorbar(mappable, cax=self._ax, orientation="horizontal")
        self._ax.set_title(title)

    def render_to_array(
        self,
        norm: Normalize,
        cm: Colormap,
        title: str = "Value",
    ) -> NDArray:
        self.render(norm, cm, title)
        self._fig.canvas.draw()
        array = np.frombuffer(
            self._fig.canvas.tostring_rgb(),  # type: ignore
            dtype=np.uint8,
        )
        w, h = self._fig.canvas.get_width_height()
        return array.reshape(h, w, -1)


def vis_birth_2d(
    ax: Axes,
    birth_fn: bd.BirthFunction,
    another_birth_fn: bd.BirthFunction | None,
    energy_max: float = 16,
    age: float = 100.0,
    initial: bool = True,
    color: str | tuple[float, float, float] = "xkcd:bluish purple",
    another_color: str | tuple[float, float, float] = "xkcd:dark aqua",
    label: str | None = None,
    another_label: str | None = None,
) -> Line2D:
    energy_max_int = int(energy_max)
    birthrate = birth_fn(
        age=jnp.ones(energy_max_int) * age,
        energy=jnp.arange(energy_max),
    )
    if another_birth_fn is None:
        lines = ax.plot(np.arange(energy_max_int), birthrate, color=color, label=label)
    else:
        another_birthrate = another_birth_fn(
            age=jnp.ones(energy_max_int) * age,
            energy=jnp.arange(energy_max),
        )
        lines = ax.plot(np.arange(energy_max_int), birthrate, color=color, label=label)
        ax.plot(
            np.arange(energy_max_int),
            another_birthrate,
            color=another_color,
            label=another_label,
        )
        ax.legend(fontsize=14.0, loc="center left", bbox_to_anchor=(0.8, 0.1))
    if initial:
        ax.grid(True, which="major")
        ax.set_xlabel("Energy $e$", fontsize=14)
        ax.yaxis.set_major_formatter("{x:.4f}")
        ax.set_ylabel("Birth prob.", fontsize=14)
    return cast(Line2D, lines[0])


def _km_formatter(x: float, _) -> str:
    if x < 1000:
        return str(x)
    elif x < 1000000:
        return f"{int(x) // 1000}K"
    else:
        return f"{int(x) // 1000000}M"


def vis_lifetime(
    ax: Axes,
    hazard_fn: bd.HazardFunction,
    energy_max: float = 16,
    n_discr: int = 101,
    initial: bool = True,
) -> Line2D:
    energy_space = np.linspace(energy_max, 0.0, n_discr)
    lifetime = np.zeros(n_discr)
    for i in range(n_discr):
        lifetime[i] = bd.compute_cumulative_survival(
            hazard_fn,
            energy=energy_space[i],
            max_age=1000000,
        )
    lines = ax.plot(energy_space, lifetime, color="xkcd:bluish purple")
    if initial:
        ax.grid(True, which="major")
        ax.set_xlabel("Energy", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_km_formatter))
        ax.set_ylabel("Expected Lifetime", fontsize=12)
    return cast(Line2D, lines[0])


def vis_expected_n_children(
    ax: Axes,
    hazard_fn: bd.HazardFunction,
    birth_fn: bd.BirthFunction,
    energy_max: float = 16,
    n_discr: int = 101,
    initial: bool = True,
) -> Line2D:
    energy_space = np.linspace(energy_max, 0.0, n_discr)
    n_children = np.zeros(n_discr)
    max_n_children = 0
    for i in range(n_discr):
        n_children[i] = bd.compute_expected_n_children(
            birth=birth_fn,
            hazard=hazard_fn,
            energy=energy_space[i],
            max_age=1000000,
        )
        max_n_children = max(max_n_children, n_children[i])
    lines = ax.plot(energy_space, n_children, color="xkcd:bluish purple")
    if initial:
        ax.grid(True, which="major")
        ax.set_xlabel("Energy", fontsize=12)
        ax.set_ylabel("Expected Num. of children", fontsize=12)
    return cast(Line2D, lines[0])


def show_params_text(
    ax: Axes,
    params: dict[str, float | int],
    columns: int = 1,
) -> list[Text]:
    params_list = list(params.items())
    n_params = len(params_list)
    unit = n_params // columns
    texts = []
    for i in range(columns):
        start = unit * i
        end = min(n_params, start + unit)
        text = ax.text(
            i * 0.9 / columns,
            1.1,
            "\n".join([f"{key}: {value:.2e}" for key, value in params_list[start:end]]),
            transform=ax.transAxes,
        )
        texts.append(text)
    return texts


def vis_hazard(
    ax: Axes3D,
    hazard_fn: bd.HazardFunction,
    age_max: int = 10000,
    energy_max: float = 16,
    hazard_max: float = 2e-4,
    n_discr: int = 101,
    method: Literal["hazard", "cumulative hazard", "survival"] = "hazard",
    initial: bool = True,
    shown_params: dict[str, float] | None = None,
) -> tuple[Poly3DCollection, Text | None]:
    age_space = jnp.linspace(0, age_max, n_discr)
    energy_space = jnp.linspace(energy_max, 0.0, n_discr)
    if method == "hazard":
        hf = hazard_fn
    elif method == "cumulative hazard":
        hf = hazard_fn.cumulative
    elif method == "survival":
        hf = hazard_fn.survival
    else:
        raise ValueError(f"Unsupported method {method}")

    death_prob = jax.vmap(lambda e: hf(age_space, jnp.ones(n_discr) * e))(energy_space)
    x, y = np.meshgrid(age_space, energy_space)
    surf = ax.plot_surface(
        x,
        y,
        death_prob,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
    )
    if initial:
        ax.set_xlim((age_max, 0.0))
        ax.set_ylim((0.0, energy_max))
        if method == "survival":
            ax.set_zlim((0.0, 1.0))
        else:
            # ax.set_zscale("log")  # type: ignore
            ax.set_zlim((1e-5, hazard_max))
            ax.zaxis.set_major_locator(ticker.LogLocator(base=100, numticks=10))
            ax.zaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x:.0e}".replace("e-0", "e-"))
            )

        ax.set_xlabel("Age $t$", fontsize=14)

        def format_age(x: float, _pos) -> str:
            del _pos
            if x > 10000:
                return f"{int(x) // 1000}K"
            else:
                return str(int(x))

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_age))
        ax.xaxis.set_ticks(np.linspace(age_max, 0.0, 5))
        ax.yaxis.set_ticks(np.linspace(0.0, energy_max, 5))
        ax.set_ylabel("Energy $e$", fontsize=14)
        if method == "hazard":
            ax.set_zlabel(
                "Hazard (Death prob.)",
                fontsize=14,
                horizontalalignment="right",
            )
        elif method == "survival":
            ax.set_zlabel("Survival prob.", fontsize=14, horizontalalignment="right")
        else:
            ax.set_zlabel(method.capitalize(), fontsize=14, horizontalalignment="right")
    if shown_params is not None:
        text = ax.text2D(
            -0.1,
            0.12,
            "\n".join([f"{key}: {value:.2e}" for key, value in shown_params.items()]),
        )
    else:
        text = None
    return surf, text


def vis_survivorship(
    ax: Axes,
    hazard_fn: bd.HazardFunction,
    age_max: int = 100000,
    energy: float = 10,
    initial: bool = True,
    color: str = "xkcd:bluish purple",
) -> Line2D:
    survival = hazard_fn.survival(jnp.arange(age_max), jnp.ones(age_max) * energy)
    lines = ax.plot(np.arange(age_max), survival, color=color)
    if initial:
        ax.grid(True, which="major")
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Survival Rate", fontsize=12)
        ax.set_ylim((0.0, 1.0))
    return cast(Line2D, lines[0])


def vis_birth(
    ax: Axes3D,
    birth_fn: bd.BirthFunction,
    age_max: int = 10000,
    energy_max: float = 16,
    n_discr: int = 101,
    initial: bool = True,
    shown_params: dict[str, float] | None = None,
) -> tuple[Poly3DCollection, Text | None]:
    age_space = jnp.linspace(0, age_max, n_discr)
    energy_space = jnp.linspace(energy_max, 0.0, n_discr)

    birth_prob = jax.vmap(lambda e: birth_fn(age_space, jnp.ones(n_discr) * e))(
        energy_space
    )
    x, y = np.meshgrid(age_space, energy_space)
    surf = ax.plot_surface(
        x,
        y,
        birth_prob,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
    )
    if initial:
        ax.set_xlim((age_max, 0.0))
        ax.set_ylim((0.0, energy_max))
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Energy", fontsize=12)
        ax.set_zlabel("Birth rate", fontsize=14, horizontalalignment="right")
    if shown_params is None:
        text = None
    else:
        text = ax.text2D(
            -0.1,
            0.08,
            "\n".join([f"{key}: {value:.2e}" for key, value in shown_params.items()]),
        )
    return surf, text
