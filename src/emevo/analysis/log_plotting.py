from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from emevo.analysis import Tree


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
    if tree is not None:
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


def plot_rewards_2d(
    ax: Axes,
    tree_df: pl.DataFrame,
    reward_df: pl.DataFrame,
    tree: Tree | None = None,
    reward_axis: str = "food",
) -> Axes:
    tr = tree_df.join(reward_df, on="unique_id")
    labels = set(tree_df["label"])
    palette = sns.color_palette("husl", len(labels))
    colors = [palette[label] for label in tr["label"]]
    ax.scatter(tr["birthtime"], tr[reward_axis], c=colors, s=5, marker="o")
    if tree is not None:

        def get_pos(ij: tuple[int, int]) -> tuple | None:
            stepi = tr.filter(pl.col("unique_id") == ij[0])
            stepj = tr.filter(pl.col("unique_id") == ij[1])
            if len(stepi) != 1 or len(stepj) != 1:
                return None
            return (
                (stepi["birthtime"].item(), stepi[reward_axis].item()),
                (stepj["birthtime"].item(), stepj[reward_axis].item()),
            )

        edge_collection = LineCollection(
            [e for e in map(get_pos, tree.all_edges()) if e is not None],
            colors="gray",
            linewidths=0.2,
            antialiaseds=(1,),
            alpha=0.4,
        )
        ax.add_collection(edge_collection)

    for label in labels:
        tr_selected = tr.filter(pl.col(f"in-label-{label}")).to_pandas()
        order = 2 if len(tr_selected) > 100 else 1
        sns.regplot(
            data=tr_selected,
            x="birthtime",
            y=reward_axis,
            scatter=False,
            color=palette[label],
            order=order,
            ax=ax,
        )

    return ax


def plot_lifehistory(
    ax: Axes,
    logdir: Path,
    log_indices: int | Iterable[int],
    unique_id: int,
    xlim: float = 480.0,
    ylim: float = 360.0,
    start_color: str = "blue",
    end_color: str = "green",
    food_color: str = "red",
) -> Axes:
    ax.set_xlim((0, xlim))
    ax.set_ylim((0, ylim))
    ax.axes.xaxis.set_visible(False)  # type: ignore
    ax.axes.yaxis.set_visible(False)  # type: ignore
    if isinstance(log_indices, int):
        log = pl.scan_parquet(logdir / f"log-{log_indices}.parquet")
        npzfile = np.load(logdir / f"state-{log_indices}.npz")
        xy = npzfile["circle_axy"][:, :, 1:]
    else:
        parquet_list = []
        xy_list = []
        for i in log_indices:
            parquet_list.append(pl.scan_parquet(logdir / f"log-{i}.parquet"))
            xy_list.append(np.load(logdir / f"state-{i}.npz")["circle_axy"][:, :, 1:])
        log = pl.concat(parquet_list)
        xy = np.concatenate(xy_list)
    indiv = log.filter(pl.col("unique_id") == unique_id).sort("step").collect()
    slot = indiv["slots"][0]
    offset = log.fetch(1)["step"][0]
    start = indiv["step"][0] - offset + 1
    end = indiv["step"][-1] - offset + 1
    xy_indiv = xy[start:end, slot]
    print(f"This agent lived {end - start + 1} steps in total from {start + offset}")
    ax.plot(xy_indiv[:, 0], xy_indiv[:, 1])
    ax.add_patch(plt.Circle(xy_indiv[0], radius=3.0, color=start_color, alpha=0.6))
    ax.add_patch(plt.Circle(xy_indiv[-1], radius=3.0, color=end_color, alpha=0.6))
    for xy in xy_indiv[indiv["got_food"][1:]]:
        ax.add_patch(plt.Circle(xy, radius=1.0, color=food_color, alpha=0.8))
    ax.set_title(f"Life history of {unique_id}")
    return ax
