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
    xy: NDArray
    is_active: NDArray


def load_agent_state(dirpath: Path, n_states: int = 10) -> AgentState:
    all_xy = []
    all_is_active = []
    for i in range(n_states):
        npzfile = np.load(dirpath / f"state-{i + 1}.npz")
        all_xy.append(npzfile["circle_axy"].astype(np.float32))
        all_is_active.append(npzfile["circle_is_active"].astype(bool))
    return AgentState(
        xy=np.concatenate(all_xy),
        is_active=np.concatenate(all_is_active),
    )


def load(logd: Path) -> tuple[AgentState, pl.DataFrame]:
    ldf = load_log(logd).with_columns(pl.col("step").alias("Step"))
    agent_state = load_agent_state(logd)
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


def compute_dxy_sim(
    agent_state: AgentState,
    stepdf: pl.DataFrame
) -> pl.DataFrame:
    def dxy_sim(start: int, end: int, slot: int) -> tuple[NDArray, NDArray]:
        # 1. Distance
        # 2. Angle
        # 3. IsActive
        xy_selected = agent_state.xy[start:end, slot]
        prey_xy = agent_state.xy[start:end, N_MAX_PREYS:]
        predator_xy = agent_state.xy[start:end, N_MAX_PREYS:]
        dxy_self = xy_selected[1:] - xy_selected[:-1]
        dxy_prey = prey_xy[1:] - prey_xy[:-1]
        dxy_predator = predator_xy[1:] - predator_xy[:-1]

    uid_list = []
    n_seeing_pred_list = []
    n_seeing_pred_back_list = []
    eaten_list = []
    for uid, slot, start, end in stepdf.iter_rows():
        if slot >= N_MAX_PREYS:  # It's predator
            continue
        if end - start < 2:
            continue
        prey_seeing, prey_seeing_back, eaten = avg_num_prey_seeing(start, end, slot)
        uid_list.append(uid)
        n_seeing_pred_list.append(prey_seeing)
        n_seeing_pred_back_list.append(prey_seeing_back)
        eaten_list.append(eaten)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Num. Obs. Pred": n_seeing_pred_list,
            "Num. Obs. Back": n_seeing_pred_back_list,
            "Eaten": eaten_list,
        }
    )
    return df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(
    logd: Path,
) -> None:
    agent_state, stepdf = load(logd)
    avgd_df = classify_agent_states(
        agent_state,
        stepdf,
        sensor_deg_in,
        sensor_deg_out,
        sensor_distance,
        mouth_deg_in,
        mouth_deg_out,
        eaten_distance,
    )
    avgd_df.write_parquet(logd / "avg-n-observing-pred.parquet")


if __name__ == "__main__":
    typer.run(main)
