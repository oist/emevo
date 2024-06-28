"""Asexual reward evolution with Circle Foraging"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray
from serde import toml

from emevo.analysis.log_plotting import load_log
from emevo.exp_utils import CfConfig, SavedPhysicsState

PROJECT_ROOT = Path(__file__).parent.parent


def _make_stats_df(profile_and_rewards_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    rdf = pl.read_parquet(profile_and_rewards_path)
    ldf = load_log(profile_and_rewards_path.parent)
    nc_df = rdf.group_by("parent").agg(n_children=pl.col("unique_id").len())
    age_df = (
        ldf.group_by("unique_id").agg(lifetime=pl.col("unique_id").count()).collect()
    )
    food_df = ldf.group_by("unique_id").agg(eaten=pl.col("n_got_food").sum()).collect()
    df = (
        rdf.join(
            nc_df, left_on="unique_id", right_on="parent", how="left", coalesce=True
        )
        .with_columns(pl.col("n_children").replace(None, 0))
        .join(age_df, left_on="unique_id", right_on="unique_id")
        .join(food_df, left_on="unique_id", right_on="unique_id")
    )
    return df, ldf


def _get_axy_arrays(path: Path, start: int, length: int):
    npzfile = np.load(path)
    caxy = npzfile["circle_axy"][start : start + length]  # (length, 200, 3)
    cact = npzfile["circle_is_active"][start : start + length]  # (length, 200)
    saxy = npzfile["static_circle_axy"][start : start + length]
    sact = npzfile["static_circle_is_active"][start : start + length]
    for i in range(length):
        c_axy_i = caxy[i][cact[i]]
        s_axy_i = saxy[i][sact[i]]
        print(c_axy_i, s_axy_i)


def _empty_list() -> list:
    return []


def main(
    profile_and_rewards_path: Path,
    starting_points: List[int] = _empty_list(),
    length: int = 100,
) -> None:
    stats_df = _make_stats_df(profile_and_rewards_path)
    log_path = profile_and_rewards_path.parent.expanduser()

    for point in starting_points:
        index = (point // 1024000) + 1
        npzfile = _get_axy_arrays(log_path / f"state-{index}.npz")


if __name__ == "__main__":
    typer.run(main)
