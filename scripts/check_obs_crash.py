import dataclasses
from pathlib import Path

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


def load(logd: Path, n_states: int = 10) -> tuple[AgentState, pl.DataFrame]:
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


def check_crash(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    obstacle_xy: NDArray,
    crash_distance: float = 25.0,
) -> pl.DataFrame:
    def crash(end: int, slot: int) -> bool:
        axy_last = agent_state.axy[end, slot]  # (3,)
        obs2prey = np.expand_dims(axy_last[1:], axis=0) - obstacle_xy
        distances = np.linalg.norm(obs2prey, axis=1)
        return bool(np.max(distances < crash_distance))

    uid_list = []
    crash_list = []
    age_list = []
    for uid, slot, start, end in stepdf.iter_rows():
        if slot >= N_MAX_PREYS:  # It's predator
            continue
        if end - start < 2:
            continue
        uid_list.append(uid)
        age = end - start + 1
        was_crashed = crash(end, slot)
        crash_list.append(was_crashed)
        age_list.append(age)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Crash": crash_list,
            "Age": age_list,
        }
    )
    return df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(
    logd: Path,
    n_states: int = 10,
    crash_distance: float = 30.0,
) -> None:
    npzfile = np.load(logd / "obstacles.npz")
    obstacle_xy = np.array(npzfile["obstacle_axy"])[:, 1:]
    agent_state, stepdf = load(logd, n_states)
    avgd_df = check_crash(
        agent_state,
        stepdf,
        obstacle_xy,
        crash_distance,
    )
    avgd_df.write_parquet(logd / "crash.parquet")


if __name__ == "__main__":
    typer.run(main)
