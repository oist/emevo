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
        )
        .collect()
    )
    return agent_state, stepdf


def classify_agent_states(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    sensor_deg_in: float = 60.0,
    sensor_deg_out: float = 300.0,
    sensor_distance: float = 224.0,  # 200 + 10 + 14
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
) -> pl.DataFrame:
    pi2 = np.pi * 2

    sensor_rad_in = np.deg2rad(sensor_deg_in)
    sensor_rad_out = np.deg2rad(sensor_deg_out)

    mouth_rad_in = np.deg2rad(mouth_deg_in)
    mouth_rad_out = np.deg2rad(mouth_deg_out)

    def avg_num_prey_seeing(
        start: int, end: int, slot: int
    ) -> tuple[NDArray, NDArray, NDArray]:
        # 1. Distance
        # 2. Angle
        # 3. IsActive
        axy_selected = agent_state.axy[start:end, slot]
        predator_axy = agent_state.axy[start:end, N_MAX_PREYS:]
        # For each xy, compute distance to predators
        expanded_axy = np.expand_dims(axy_selected[:, 1:], axis=1)
        xydiff = predator_axy[:, :, 1:] - expanded_axy
        distances = np.linalg.norm(xydiff, axis=2)
        # Compute angle
        expanded_angle = np.expand_dims(axy_selected[:, 0], axis=1)
        dxy = predator_axy[:, :, 1:] - expanded_axy
        rel_angle = np.arctan2(dxy[:, :, 1], dxy[:, :, 0])
        pred_pos_angle = (rel_angle - expanded_angle + pi2) % pi2
        in_prey_sensor = (pred_pos_angle <= sensor_rad_in) | (
            sensor_rad_out <= pred_pos_angle
        )
        rel_angle_prey = (predator_axy[:, :, 0] - expanded_angle + pi2) % pi2
        is_pred_face_to_prey = (rel_angle_prey <= sensor_rad_in) | (
            sensor_rad_out <= rel_angle_prey
        )
        # Accumulate
        is_pred_active = agent_state.is_active[start:end, N_MAX_PREYS:]
        in_sensor_dist = distances < sensor_distance
        prey_seeing = (in_sensor_dist & in_prey_sensor) & is_pred_active
        prey_seeing_back = prey_seeing & (~is_pred_face_to_prey)
        # Eaten?
        can_pred_eat_prey = (rel_angle_prey[-1] <= mouth_rad_in) | (
            mouth_rad_out <= rel_angle_prey[-1]
        )  # (N_PRED,)
        eaten = np.max(can_pred_eat_prey[-1] & (distances[-1] < eaten_distance))
        return (
            np.mean(np.sum(prey_seeing, axis=1)),
            np.mean(np.sum(prey_seeing_back, axis=1)),
            eaten,
        )

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
        if uid > 100:
            break
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
    sensor_deg_in: float = 60.0,
    sensor_deg_out: float = 300.0,
    sensor_distance: float = 224.0,
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
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
