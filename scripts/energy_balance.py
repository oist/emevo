from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10, bin_length: int = 1000) -> None:
    age_df = ldf.with_columns(
        (pl.col("step") - pl.col("step").min().over("unique_id")).alias("age")
    )
    balance_df = (
        age_df.group_by("unique_id", ((pl.col("age") // bin_length) * bin_length).alias("age_bin"))
        .agg(
            (pl.col("energy_gain").sum() / pl.col("consumed_energy").sum()).alias("Energy Balance")
        )
        .sort("unique_id", "age_bin")
        .collect()
    )
    balance_df.write_parquet(logd / f"balance_bin{bin_length}.parquet")


if __name__ == "__main__":
    typer.run(main)
