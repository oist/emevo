from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log


def main(logd: Path, n_states: int = 10, span: int = 100) -> None:
    ldf = load_log(logd, last_idx=n_states).with_columns(
        (pl.col("step") // span * span).alias("Step")
    )
    pop_df = ldf.group_by("Step").agg(pl.col("unique_id").len().alias("Population"))
    pop_df.collect().write_parquet(logd / "popl.parquet")


if __name__ == "__main__":
    typer.run(main)
