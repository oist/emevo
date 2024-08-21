import importlib
from pathlib import Path
from typing import List

import matplotlib as mpl
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from serde import toml

from emevo.exp_utils import BDConfig
from emevo.plotting import vis_birth_2d


def compare_birth_models(config_list: List[Path], energy_max: float = 40) -> None:
    if importlib.util.find_spec("PySide6") is not None:  # type: ignore
        mpl.use("QtAgg")
    else:
        mpl.use("TkAgg")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Compare birth functions", fontsize=16)
    palette = sns.color_palette("husl", len(config_list))
    for config, color in zip(config_list, palette):
        with config.open("r") as f:
            bd_config = toml.from_toml(BDConfig, f.read())
        birth_model, _ = bd_config.load_models()
        vis_birth_2d(
            ax,
            birth_fn=birth_model,
            energy_max=energy_max,
            initial=True,
            color=color,
            label=config.stem,
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    typer.run(compare_birth_models)
