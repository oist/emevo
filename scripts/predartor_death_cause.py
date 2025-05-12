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


def load(logd: Path) -> tuple[AgentState, pl.DataFrame]:
    ldf = load_log(logd).with_columns(pl.col("step").alias("Step"))
    agent_state = load_agent_state(logd)
    stepdf = (
        ldf.group_by("unique_id")
        .agg(
            pl.col("slots").first(),
            pl.col("step").min().alias("start"),
            pl.col("step").max().alias("end"),
            pl.col("energy").last().alias("last_energy"),
        )
        .collect()
    )
    return agent_state, stepdf


def classify_death_cause(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
) -> pl.DataFrame:
    pi2 = np.pi * 2

    mouth_rad_in = np.deg2rad(mouth_deg_in)
    mouth_rad_out = np.deg2rad(mouth_deg_out)

    def death_cause(start: int, end: int, slot: int, last_energy: float) -> str:
        axy_selected = agent_state.axy[start:end, slot]
        predator_axy = agent_state.axy[start:end, N_MAX_PREYS:]
        # For each xy, compute distance to predators
        expanded_axy = np.expand_dims(axy_selected[:, 1:], axis=1)
        xydiff = predator_axy[:, :, 1:] - expanded_axy
        distances = np.linalg.norm(xydiff, axis=2)
        # Compute angle
        expanded_angle = np.expand_dims(axy_selected[:, 0], axis=1)
        rel_angle_prey = (predator_axy[:, :, 0] - expanded_angle + pi2) % pi2
        # Eaten?
        can_pred_eat_prey = (rel_angle_prey[-1] <= mouth_rad_in) | (
            mouth_rad_out <= rel_angle_prey[-1]
        )  # (N_PRED,)
        eaten = np.max(can_pred_eat_prey[-1] & (distances[-1] < eaten_distance))
        if eaten:
            return "Eaten"
        elif last_energy < 0.1:
            return "Hunger"
        else:
            return "Age"

    uid_list = []
    death_cause_list = []
    for uid, slot, start, end, last_energy in stepdf.iter_rows():
        if slot >= N_MAX_PREYS:  # It's predator
            continue
        if end - start < 2:
            continue
        uid_list.append(uid)
        death_cause_list.append(death_cause(start, end, slot, last_energy))
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Death Cause": death_cause_list,
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
    avgd_df = classify_death_cause(
        agent_state,
        stepdf,
        mouth_deg_in,
        mouth_deg_out,
        eaten_distance,
    )
    avgd_df.write_parquet(logd / "death_cause.parquet")


if __name__ == "__main__":
    typer.run(main)
