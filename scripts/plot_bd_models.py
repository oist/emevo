import importlib
from pathlib import Path
from typing import cast

import matplotlib as mpl
import typer
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from serde import toml

from emevo.exp_utils import BDConfig
from emevo.plotting import (
    show_params_text,
    vis_birth,
    vis_birth_2d,
    vis_expected_n_children,
    vis_hazard,
    vis_lifetime,
    vis_survivorship,
)


def plot_bd_models(
    config: Path = Path("config/bd/20230530-a035-e020.toml"),
    another_config: Path | None = None,
    age_max: int = 200000,
    energy_max: float = 40,
    hazard_max: float = 2e-4,
    survivorship_energy: float = 10.0,
    n_discr: int = 100,
    yes: bool = typer.Option(False, help="Skip all yes/no prompts"),
    horizontal: bool = typer.Option(
        False,
        help="Use horizontal order for multiple figures",
    ),
    noparam: bool = typer.Option(False, help="Don't show parameters"),
    nolifespan: bool = typer.Option(False, help="Don't show lifespan"),
    simpletitle: bool = typer.Option(False, help="Make title simple"),
    birth2d: bool = typer.Option(False, help="Make 2D plot for birth rate"),
    all2d: bool = typer.Option(
        False,
        help="Plot birth, lifetime, and n. children on the same fig",
    ),
) -> None:
    if importlib.util.find_spec("PySide6") is not None:  # type: ignore
        mpl.use("QtAgg")
    else:
        mpl.use("TkAgg")

    with config.open("r") as f:
        bd_config = toml.from_toml(BDConfig, f.read())

    birth_model, hazard_model = bd_config.load_models()

    if another_config is not None:
        with another_config.open("r") as f:
            another_bd_config = toml.from_toml(BDConfig, f.read())
            another_birth_model, another_hazard_model = another_bd_config.load_models()
    else:
        another_birth_model, another_hazard_model = None, None

    if yes or typer.confirm("Plot hazard model?"):
        if horizontal:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
        else:
            fig = plt.figure(figsize=(5, 10))
            ax1 = fig.add_subplot(211, projection="3d")
            ax2 = fig.add_subplot(212, projection="3d")
        if simpletitle:
            ax1.set_title("Hazard function $h(t, e)$", fontsize=18)  # type: ignore
            ax2.set_title("Survival function $s_e(t)$", fontsize=18)  # type: ignore
        else:
            ax1.set_title(f"{type(hazard_model).__name__} Hazard function")  # type: ignore
            ax2.set_title(f"{type(hazard_model).__name__} Survival function")  # type: ignore
        vis_hazard(
            cast(Axes3D, ax1),
            hazard_fn=hazard_model,
            age_max=age_max,
            energy_max=energy_max,
            hazard_max=hazard_max,
            n_discr=n_discr,
            method="hazard",
            shown_params=None if noparam else bd_config.hazard_params,
        )
        vis_hazard(
            cast(Axes3D, ax2),
            hazard_fn=hazard_model,
            age_max=age_max,
            energy_max=energy_max,
            n_discr=n_discr,
            method="survival",
        )
        plt.show()

    if yes or typer.confirm("Plot birth model?"):
        fig = plt.figure(figsize=(8, 4))
        if birth2d:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")
        if simpletitle:
            ax.set_title("Birth function", fontsize=16)
        else:
            ax.set_title(f"{type(birth_model).__name__} Birth function", fontsize=14)
        if birth2d:
            vis_birth_2d(
                ax,
                birth_fn=birth_model,
                another_birth_fn=another_birth_model,
                energy_max=energy_max,
                initial=True,
                label="Prey",
                another_label="Predator",
            )
        else:
            vis_birth(
                cast(Axes3D, ax),
                birth_fn=birth_model,
                age_max=age_max,
                energy_max=energy_max,
                n_discr=n_discr,
                initial=True,
                shown_params=None if noparam else bd_config.birth_params,
            )
        plt.show()

    if yes or typer.confirm("Plot survivor ship curve?"):
        fig = plt.figure(figsize=(5, 10))
        ax = fig.add_subplot(111)
        tname = type(birth_model).__name__
        ax.set_title(f"{tname} Survivor ship when energy={survivorship_energy}")
        vis_survivorship(ax=ax, hazard_fn=hazard_model, age_max=age_max, initial=True)
        plt.show()

    if yes or typer.confirm("Plot expected num. of children?"):
        if nolifespan:
            fig = plt.figure(figsize=(6, 4))
            ax1 = None
            ax2 = fig.add_subplot(111)
        elif horizontal:
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            fig = plt.figure(figsize=(6, 10))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        if simpletitle:
            name = ""
        else:
            name = f"{type(hazard_model).__name__} & {type(birth_model).__name__} "
        if ax1 is not None:
            ax1.set_title(f"{name}Expected Lifetime", fontsize=14)
            vis_lifetime(
                ax1,
                hazard_fn=hazard_model,
                energy_max=energy_max,
                n_discr=n_discr,
            )
            if not noparam:
                params = bd_config.hazard_params | {
                    f"birth_{key}": value
                    for key, value in bd_config.birth_params.items()
                }
                show_params_text(ax1, params, columns=2)

        ax2.set_title(f"{name}Expected Num. of children", fontsize=14)
        vis_expected_n_children(
            ax2,
            birth_fn=birth_model,
            hazard_fn=hazard_model,
            energy_max=energy_max,
            n_discr=n_discr,
        )
        plt.show()

    if all2d and (yes or typer.confirm("Plot birth and n.children on the same fig?")):
        fig = plt.figure(figsize=(15, 5))
        fig.tight_layout()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_title("Birth function", fontsize=14)
        vis_birth_2d(
            ax1,
            birth_fn=birth_model,
            energy_max=energy_max,
            initial=True,
        )
        ax2.set_title("Expected Lifetime", fontsize=14)
        vis_lifetime(
            ax2,
            hazard_fn=hazard_model,
            energy_max=energy_max,
            n_discr=n_discr,
        )
        ax3.set_title("Expected Num. of children", fontsize=14)
        vis_expected_n_children(
            ax3,
            birth_fn=birth_model,
            hazard_fn=hazard_model,
            energy_max=energy_max,
            n_discr=n_discr,
        )
        plt.show()


if __name__ == "__main__":
    typer.run(plot_bd_models)
