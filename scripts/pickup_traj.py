import dataclasses
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray

from emevo.analysis.log_plotting import load_log

N_MAX_PREYS = 450


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
    unique_id: list[int],
    n_states: int = 10,
) -> tuple[AgentState, pl.DataFrame]:
    agent_state = load_agent_state(logd, n_states=n_states)
    if (logd / "age.parquet").exists():
        return agent_state, pl.read_parquet(logd / "age.parquet").select(
            "unique_id", "slots", "start", "end"
        )
    ldf = (
        load_log(logd, last_idx=n_states)
        .with_columns(pl.col("step").alias("Step"))
        .filter(pl.col("unique_id").is_in(unique_id))
    )
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


def pickup_traj(agent_state: AgentState, stepdf: pl.DataFrame) -> pl.DataFrame:
    uid_list = []
    step_list = []
    x_list = []
    y_list = []
    for uid, slot, start, end in stepdf.iter_rows():
        if slot >= N_MAX_PREYS:  # It's predator
            continue
        if end - start < 2:
            continue
        uid_list.append(uid)
        for i, (_, x, y) in enumerate(agent_state.axy[start:end, slot]):
            x_list.append(x)
            y_list.append(y)
            uid_list.append(uid)
            step_list.append(i + start)
    return pl.from_dict(
        {
            "unique_id": uid_list,
            "x": x_list,
            "y_id": y_list,
            "step_id": step_list,
        }
    )


def main(
    logd: Path,
    unique_id: Annotated[list[int], typer.Argument(help="One or more unique IDs.")],
    n_states: int = 10,
) -> None:
    agent_state, stepdf = load(logd, unique_id, n_states)
    traj_df = pickup_traj(agent_state, stepdf)
    traj_df.write_parquet(logd / "pickup-traj.parquet")


if __name__ == "__main__":
    typer.run(main)
