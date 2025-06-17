from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10) -> None:
    logdf = load_log(logd, last_idx=n_states).with_columns(pl.col("step").alias("Step")).sort("Step")
    df = (
        logdf.group_by("unique_id")
        .agg(pl.col("energy").first().alias("Initial Energy"))
        .collect()
    )
    df.write_parquet(logd / "birth_energy.parquet")


if __name__ == "__main__":
    typer.run(main)
