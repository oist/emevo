from pathlib import Path

import polars as pl
import typer


def log_iter(basepath: Path, n_states: int = 10):
    for i in range(n_states):
        idx = i + 1
        logpath = basepath.joinpath(f"log-{idx}.parquet").expanduser()
        if logpath.exists():
            yield pl.scan_parquet(logpath)


def main(logd: Path, predator_slot: int = 450, n_states: int = 10) -> None:
    popl_df_list = []
    for log in log_iter(logd, n_states):
        ldf = log.with_columns(
            pl.col("step").alias("Step"),
            pl.when(pl.col("slots") < predator_slot)
            .then(pl.lit("prey"))
            .otherwise(pl.lit("predator"))
            .alias("Species"),
        )
        popl_df = ldf.group_by("Step", "Species").agg(
            pl.len().alias("Population"),
        )
        popl_df_list.append(popl_df.collect())
    df = pl.concat(popl_df_list)
    df.write_parquet(logd / "popl.parquet")


if __name__ == "__main__":
    typer.run(main)
