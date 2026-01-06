from pathlib import Path

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray

from emevo.analysis.log_plotting import load_log


def load_xy(dirpath: Path, n_states: int = 10) -> NDArray:
    all_xy = []
    for i in range(n_states):
        npzfile = np.load(dirpath / f"state-{i + 1}.npz")
        circle_axy = npzfile["circle_axy"].astype(np.float32)
        all_xy.append(circle_axy[:, :, 1:])
    return np.concatenate(all_xy)


def load(logd: Path) -> tuple[NDArray, pl.DataFrame]:
    ldf = load_log(logd).with_columns(pl.col("step").alias("Step"))
    xy = load_xy(logd)
    stepdf = (
        ldf.group_by("unique_id")
        .agg(
            pl.col("slots").first(),
            pl.col("step").min().alias("start"),
            pl.col("step").max().alias("end"),
        )
        .collect()
    )
    return xy, stepdf


def compute_avg_moved(xy: NDArray, stepdf: pl.DataFrame, unit: int) -> pl.DataFrame:
    def avgd(start: int, end: int, slot: int) -> NDArray:
        xy_selected = xy[start:end, slot]
        xy0 = xy_selected[:-1]
        xy1 = xy_selected[1:]
        return np.mean(np.linalg.norm(xy0 - xy1, axis=1))

    uid_list = []
    time_list = []
    avgd_list = []
    for uid, slot, start, end in stepdf.iter_rows():
        length = end - start
        if length < 4:
            continue
        for i in range(length // unit + 1):
            current = start + 1 + unit * i
            if end - current < 3:
                break
            avgd_ = avgd(current, min(current + unit, end), slot)
            current += unit
            uid_list.append(uid)
            avgd_list.append(avgd_)
            time_list.append((i + 1) * unit)
    avgd_df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Avg. Distances": avgd_list,
            "Step": time_list,
        }
    )
    return avgd_df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(logd: Path, unit: int = 100) -> None:
    xy, stepdf = load(logd)
    avgd_df = compute_avg_moved(xy, stepdf, unit)
    avgd_df.write_parquet(logd / f"avgd_per_{unit}step.parquet")


if __name__ == "__main__":
    typer.run(main)
