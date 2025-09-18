from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10) -> None:
    logdf = load_log(logd, last_idx=n_states)
    df = logdf.group_by("unique_id").agg(
        pl.col("consumed_energy").mean().alias("average_ec")
    )
    df.collect().write_parquet(logd / "average_ec.parquet")


if __name__ == "__main__":
    typer.run(main)
