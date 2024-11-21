"""Value visualization"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def plot_values(
    values: NDArray,
    names: list[str],
    images: list[NDArray],
    title: str = "Value Functions",
    fig_unit: float = 6.0,
    show: bool = True,
) -> None:
    # Validate inputs
    assert values.ndim == 2, "values must be a 2D array"
    assert (
        len(names) == values.shape[1]
    ), "Number of names must match number of columns in values"

    observations = [f"Observation {i+1}" for i in range(values.shape[0])]
    n_images = len(images)
    figsize = fig_unit * n_images, fig_unit * 2
    fig = plt.figure(figsize=figsize)

    # Create a grid specification
    gs = fig.add_gridspec(
        2,
        n_images,
        width_ratios=[1] * n_images,
        height_ratios=[1, 1],
    )
    ax = fig.add_subplot(gs[0, :n_images])

    # Plot bars for each observation
    obs_names = []
    net_names = []
    value_list = []
    for i, obs_name in enumerate(observations):
        for j, net_name in enumerate(names):
            obs_names.append(obs_name)
            net_names.append(net_name)
            value_list.append(values[i][j])

    df = pd.DataFrame.from_dict(
        {"Obs": obs_names, "Indiv": net_names, "Values": value_list}
    )
    sns.barplot(df, x="Obs", y="Values", hue="Indiv", ax=ax)
    ax.set_title(title)
    for i, image in enumerate(images):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(image)
        ax.set_title(f"{observations[i]}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if show:
        plt.show()
