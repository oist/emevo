"""Asexual reward evolution with Circle Foraging"""

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import typer

from emevo.analysis.log_plotting import load_log

PROJECT_ROOT = Path(__file__).parent.parent


def _make_stats_df(profile_and_rewards_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    rdf = pl.read_parquet(profile_and_rewards_path)
    ldf = load_log(profile_and_rewards_path.parent).cast({"unique_id": pl.Int64})
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


def _agg_df(
    path: Path,
    start: int,
    length: int,
    ldf: pl.DataFrame,
    ldf_offset: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    npzfile = np.load(path)
    caxy = npzfile["circle_axy"][start : start + length]  # (length, 200, 3)
    cact = npzfile["circle_is_active"][start : start + length]  # (length, 200)
    saxy = npzfile["static_circle_axy"][start : start + length]
    sact = npzfile["static_circle_is_active"][start : start + length]
    slabel = npzfile["static_circle_label"][start : start + length]
    cx_list, cy_list, ca_list = [], [], []
    sx_list, sy_list, slab_list = [], [], []
    uniqueid_list, c_nsteps_list, s_nsteps_list = [], [], []
    for i in range(length):
        active_slots = np.nonzero(cact[i])
        caxy_i = caxy[i][active_slots]
        saxy_i = saxy[i][sact[i]]

        sx_list.append(saxy_i[:, 1])
        sy_list.append(saxy_i[:, 2])
        slab_list.append(slabel[i][sact[i]])

        ca_list.append(caxy_i[:, 0])
        cx_list.append(caxy_i[:, 1])
        cy_list.append(caxy_i[:, 2])
        df = ldf.filter(pl.col("step") == ldf_offset + start + i).sort("slots")
        if len(df) != len(caxy_i):
            warnings.warn(
                "Number of active agents doesn't match"
                + f"State: {len(saxy_i)} Log: {len(df)}"
                + f"at step {ldf_offset + start + i}",
                stacklevel=1,
            )
            df = df.unique(subset="unique_id", keep="first")
            df = df.filter(((pl.col("unique_id") == 0) & (pl.col("slots") != 0)).not_())
        uniqueid_list.append(df["unique_id"])
        # Num. steps
        c_nsteps_list.append(df["step"])
        s_nsteps_list.append([ldf_offset + start + i] * len(saxy_i))

    cxy_df = pl.DataFrame(
        {
            "angle": np.concatenate(ca_list),
            "x": np.concatenate(cx_list),
            "y": np.concatenate(cy_list),
            "unique_id": pl.concat(uniqueid_list),
            "nsteps": pl.concat(c_nsteps_list),
        }
    )
    sxy_df = pl.DataFrame(
        {
            "x": np.concatenate(sx_list),
            "y": np.concatenate(sy_list),
            "label": np.concatenate(slab_list),
            "nsteps": np.concatenate(s_nsteps_list),
        }
    )
    return cxy_df, sxy_df


def main(
    profile_and_rewards_path: Path,
    starting_points: List[int],
    write_dir: Optional[Path] = None,
    length: int = 100,
) -> None:
    if write_dir is None:
        write_dir = Path("saved-web-data")

    stats_df, ldf = _make_stats_df(profile_and_rewards_path)
    stats_df.write_parquet(write_dir / "stats.parqut", compression="snappy")

    log_path = profile_and_rewards_path.parent.expanduser()

    for point in starting_points:
        index = point // 1024000
        ldfi = ldf.filter(
            (pl.col("step") >= point) & (pl.col("step") < point + length)
        ).collect()  # Offloading here for speedup
        cxy_df, sxy_df = _agg_df(
            log_path / f"state-{index + 1}.npz",
            point - index * 1024000,
            length,
            ldfi,
            index * 1024000,
        )
        cxy_df.write_parquet(
            write_dir / f"saved_cpos-{point}.parqut",
            compression="snappy",
        )
        sxy_df.write_parquet(
            write_dir / f"saved_spos-{point}.parqut",
            compression="snappy",
        )


if __name__ == "__main__":
    typer.run(main)
