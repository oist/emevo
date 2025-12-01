import dataclasses
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray

from emevo.analysis.log_plotting import load_log


@dataclasses.dataclass
class AgentState:
    axy: NDArray
    is_active: NDArray


def load_agent_state(dirpath: Path, n_states: int = 10) -> AgentState:
    all_axy = []
    all_is_active = []
    for i in range(n_states):
        npzfile = np.load(dirpath / f"state-{i + 1}.npz")
        all_axy.append(npzfile["circle_axy"].astype(np.float32))
        all_is_active.append(npzfile["circle_is_active"].astype(bool))
    return AgentState(
        axy=np.concatenate(all_axy),
        is_active=np.concatenate(all_is_active),
    )


def load(
    logd: Path,
    n_states: int = 10,
) -> tuple[AgentState, pl.DataFrame]:
    agent_state = load_agent_state(logd, n_states=n_states)
    if (logd / "age.parquet").exists():
        return agent_state, pl.read_parquet(logd / "age.parquet").select(
            "unique_id", "slots", "start", "end"
        )
    ldf = load_log(logd, last_idx=n_states).with_columns(pl.col("step").alias("Step"))
    stepdf = (
        ldf.group_by("unique_id")
        .agg(
            pl.col("slots").first(),
            pl.col("step").min().alias("start"),
            pl.col("step").max().alias("end"),
        )
        .collect()
    )
    return agent_state, stepdf


def assign_range(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    x_grid: int,
    y_grid: int,
    max_x: int,
    max_y: int,
) -> pl.DataFrame:
    uid_list = []
    t_list = []
    range_dict = defaultdict(list)
    bins_x = np.arange(0, max_x + 1, x_grid)
    bins_y = np.arange(0, max_y + 1, y_grid)
    for uid, slot, start, end in stepdf.iter_rows():
        if end - start < 2:
            pass
        # Skip time 0
        xy = agent_state.axy[start:end, slot][1:, 1:]
        t = xy.shape[0]
        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[bins_x, bins_y])
        for x, y in itertools.product(range(max_x // x_grid), range(max_y // y_grid)):
            range_dict[f"({x + 1}, {y + 1})"].append(hist[x, y] / t)
        uid_list.append(uid)
        t_list.append(t)
    return pl.from_dict(
        {
            "unique_id": uid_list,
            "t": t_list,
            **range_dict,
        }
    )


def main(
    logd: Path,
    n_states: int = 10,
    x_grid: int = 60,
    y_grid: int = 60,
    max_x: int = 480,
    max_y: int = 480,
) -> None:
    agent_state, stepdf = load(logd, n_states)
    df = assign_range(agent_state, stepdf, x_grid, y_grid, max_x, max_y)
    df.write_parquet(logd / "xyrange.parquet")


if __name__ == "__main__":
    typer.run(main)
