from pathlib import Path

import polars as pl
import typer


def log_iter(basepath: Path, n_states: int = 10):
    for i in range(n_states):
        idx = i + 1
        logpath = basepath.joinpath(f"log-{idx}.parquet").expanduser()
        if logpath.exists():
            yield pl.scan_parquet(logpath)


def main(logd: Path, n_states: int = 10) -> None:
    max_step = 0
    for log in log_iter(logd, n_states):
        maxstep_df = log.select(pl.col("step").max()).collect()
        max_step = max(max_step, maxstep_df.item())
    print(max_step)


if __name__ == "__main__":
    typer.run(main)
