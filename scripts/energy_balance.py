from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10, bin_length: int = 1000) -> None:
    logdf = load_log(logd, last_idx=n_states)
    age_df = logdf.with_columns(
        (pl.col("step") - pl.col("step").min().over("unique_id")).alias("age"),
        pl.col("energy").diff().alias("energy_delta"),
    )
    age_ed_df = age_df.with_columns(
        pl.when(pl.col("age") == 0)
        .then(pl.col("energy_gain") - pl.col("consumed_energy"))
        .otherwise(pl.col("energy_delta"))
        .alias("energy_delta"),
    )
    balance_df = (
        age_ed_df.group_by(
            "unique_id", ((pl.col("age") // bin_length) * bin_length).alias("age_bin")
        )
        .agg(
            pl.col("energy_gain").mean(),
            pl.col("consumed_energy").mean(),
            (pl.col("energy_gain") - pl.col("consumed_energy") - pl.col("energy_delta"))
            .mean()
            .alias("repr_energy_consumption"),
            pl.col("rewards").mean(),
        )
        .sort("unique_id", "age_bin")
        .collect()
    )
    balance_df.write_parquet(logd / f"balance_bin{bin_length}.parquet")


if __name__ == "__main__":
    typer.run(main)
