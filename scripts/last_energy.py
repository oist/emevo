from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10) -> None:
    logdf = load_log(logd, last_idx=n_states).with_columns(pl.col("step").alias("Step"))
    df = (
        logdf.sort("Step")
        .group_by("unique_id")
        .agg(
            pl.col("energy").last().alias("Last Energy"),
            pl.col("unique_id").len().alias("Lifetime"),
        )
    ).collect()
    df.write_parquet(logd / "last-energy.parquet")


if __name__ == "__main__":
    typer.run(main)
