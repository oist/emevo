from os import PathLike
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from emevo.analysis import Tree
from emevo.exp_utils import SavedPhysicsState


def load_log(pathlike: PathLike, last_idx: int = 10) -> pl.LazyFrame:
    if isinstance(pathlike, Path):
        path = pathlike
    else:
        path = Path(pathlike)
    parquets = []
    for idx in range(1, last_idx + 1):
        logpath = path.joinpath(f"log-{idx}.parquet").expanduser()
        if logpath.exists():
            parquets.append(pl.scan_parquet(logpath))
    return pl.concat(parquets)


def plot_rewards_3d(
    ax: Axes3D,
    reward_df: pl.DataFrame,
    tree_df: pl.DataFrame,
    reward_prefix_1: str = "w",
    reward_prefix_2: str = "alpha",
    tree: Tree | None = None,
    reward_axis: str = "food",
) -> Axes3D:
    tr = tree_df.join(reward_df, on="unique_id")
    labels = set(tree_df["label"])
    palette = sns.color_palette("husl", len(labels))
    r1, r2 = f"{reward_prefix_1}_{reward_axis}", f"{reward_prefix_2}_{reward_axis}"
    colors = [palette[label] for label in tree_df["label"]]
    scatter = ax.scatter(tr[r1], tr[r2], tr["birthtime"], c=colors, s=5, marker="o")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_xlabel(r1)
    ax.set_ylabel(r2)
    ax.set_zlabel("Birth Step")
    if tree != None:
        x, y, z = scatter._offsets3d  # type: ignore
        x = x.data
        y = y.data

        def get_pos(ij: tuple[int, int]) -> tuple | None:
            i, j = ij[0] - 1, ij[1] - 1
            if i >= len(x) or j >= len(x):
                return None
            return ((x[i], y[i], z[i]), (x[j], y[j], z[j]))

        edge_collection = Line3DCollection(
            [e for e in map(get_pos, tree.all_edges()) if e is not None],
            colors="gray",
            linewidths=0.1,
            alpha=0.4,
        )
        ax.add_collection(edge_collection)
    return ax


def plot_rewards(
    ax: Axes,
    reward_df: pl.DataFrame,
    tree_df: pl.DataFrame,
    tree: Tree | None = None,
    reward_axis: str = "food",
) -> Axes:
    tr = tree_df.join(reward_df, on="index")
    labels = set(tree_df["label"])
    palette = sns.color_palette("husl", len(labels))
    sns.scatterplot(
        data=tr,
        x="birth-step",
        y=reward_axis,
        hue="label",
        palette=palette,
        ax=ax,
        legend=False,
    )
    if tree != None:

        def get_pos(ij: tuple[int, int]) -> tuple | None:
            stepi = tr.filter(pl.col("index") == ij[0])
            stepj = tr.filter(pl.col("index") == ij[1])
            if len(stepi) != 1 or len(stepj) != 1:
                return None
            return (
                (stepi["birthtime"].item(), stepi[reward_axis].item()),
                (stepj["birthtime"].item(), stepj[reward_axis].item()),
            )

        edge_collection = LineCollection(
            [e for e in map(get_pos, tree.all_edges()) if e is not None],
            colors="gray",
            linewidths=0.5,
            antialiaseds=(1,),
            alpha=0.6,
        )
        ax.add_collection(edge_collection)
    return ax


def plot_lifehistory(
    ax: Axes,
    phys_state: SavedPhysicsState,
    slot: int,
    start: int,
    end: int,
    xlim: float = 480.0,
    ylim: float = 360.0,
) -> None:
    assert start < end
    axy = np.array(phys_state.circle_axy[start:end, slot])
    x = axy[1]
    y = axy[2]
    ax.plot(x, y)
    return ax
