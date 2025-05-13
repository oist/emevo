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
    ldf = load_log(logd, last_idx=n_states).with_columns(pl.col("step").alias("Step"))
    agent_state = load_agent_state(logd, n_states=n_states)
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


def check_eaten(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
) -> pl.DataFrame:
    pi2 = np.pi * 2

    mouth_rad_in = np.deg2rad(mouth_deg_in)
    mouth_rad_out = np.deg2rad(mouth_deg_out)

    def eaten(end: int, slot: int) -> bool:
        axy_last = agent_state.axy[end, slot]  # (3,)
        predator_axy_last = agent_state.axy[end, N_MAX_PREYS:]  # (M, 3)
        # Compute the distances to all predators when the prey dies
        xydiff = predator_axy_last[:, 1:] - np.expand_dims(axy_last[1:], axis=0)
        distances = np.linalg.norm(xydiff, axis=1)
        # Compute angle
        rel_angle = (predator_axy_last[:, 0] - axy_last[0] + pi2) % pi2
        # Eaten?
        can_pred_eat_prey = (rel_angle <= mouth_rad_in) | (mouth_rad_out <= rel_angle)
        return bool(np.max(can_pred_eat_prey & (distances < eaten_distance)))

    uid_list = []
    eaten_list = []
    age_list = []
    for uid, slot, start, end, last_energy in stepdf.iter_rows():
        if slot >= N_MAX_PREYS:  # It's predator
            continue
        if end - start < 2:
            continue
        uid_list.append(uid)
        age = end - start + 1
        eaten_list.append(eaten(end, slot))
        age_list.append(age)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Eaten": eaten_list,
            "Age": age_list,
        }
    )
    return df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(
    logd: Path,
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
) -> None:
    agent_state, stepdf = load(logd)
    avgd_df = check_eaten(
        agent_state,
        stepdf,
        mouth_deg_in,
        mouth_deg_out,
        eaten_distance,
    )
    avgd_df.write_parquet(logd / "eaten.parquet")


if __name__ == "__main__":
    typer.run(main)
