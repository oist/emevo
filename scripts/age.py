from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10) -> None:
    ldf = load_log(logd, last_idx=n_states).with_columns(pl.col("step").alias("Step"))
    stepdf = ldf.group_by("unique_id").agg(
        pl.col("slots").first(),
        pl.col("step").min().alias("start"),
        pl.col("step").max().alias("end"),
    )
    agedf = stepdf.with_columns((pl.col("end") - pl.col("start") + 1).alias("Age"))
    agedf.collect().write_parquet(logd / "age.parquet")


if __name__ == "__main__":
    typer.run(main)
