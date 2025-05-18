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


def check_eaten(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    sensor_deg_in: float = 60.0,
    sensor_deg_out: float = 300.0,
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
) -> pl.DataFrame:
    pi2 = np.pi * 2

    sensor_rad_in = np.deg2rad(sensor_deg_in)
    sensor_rad_out = np.deg2rad(sensor_deg_out)

    mouth_rad_in = np.deg2rad(mouth_deg_in)
    mouth_rad_out = np.deg2rad(mouth_deg_out)

    def eaten(end: int, slot: int) -> tuple[bool, bool]:
        axy_last = agent_state.axy[end, slot]  # (3,)
        predator_axy_last = agent_state.axy[end, N_MAX_PREYS:]  # (M, 3)
        # Compute the distances to all predators when the prey dies
        pred2prey = np.expand_dims(axy_last[1:], axis=0) - predator_axy_last[:, 1:]
        distances = np.linalg.norm(pred2prey, axis=1)
        # Compute relative angle
        angle_pred2prey = np.arctan2(pred2prey[:, 1], pred2prey[:, 0])
        rel_angle_pred = (angle_pred2prey - predator_axy_last[:, 0] + pi2) % pi2
        # Eaten?
        in_mouth_range = (rel_angle_pred <= mouth_rad_in) | (
            mouth_rad_out <= rel_angle_pred
        )
        can_pred_eat_prey = in_mouth_range & agent_state.is_active[end, N_MAX_PREYS:]
        was_eaten = can_pred_eat_prey & (distances < eaten_distance)
        # Compute relative angle (from prey)
        angle_prey2pred = np.arctan2(-pred2prey[:, 1], -pred2prey[:, 0])
        rel_angle_prey = (angle_prey2pred - axy_last[0] + pi2) % pi2
        is_pred_in_sensor = (rel_angle_prey <= sensor_rad_in) | (
            sensor_rad_out <= rel_angle_prey
        )
        eaten_and_in_sensor = was_eaten & is_pred_in_sensor
        return bool(np.max(was_eaten)), bool(np.max(eaten_and_in_sensor))

    uid_list = []
    eaten_list = []
    in_sensor_list = []
    age_list = []
    for uid, slot, start, end in stepdf.iter_rows():
        if slot >= N_MAX_PREYS:  # It's predator
            continue
        if end - start < 2:
            continue
        uid_list.append(uid)
        age = end - start + 1
        was_eaten, in_sensor = eaten(end, slot)
        eaten_list.append(was_eaten)
        in_sensor_list.append(in_sensor)
        age_list.append(age)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Eaten": eaten_list,
            "Eaten (in sensor)": in_sensor_list,
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
    sensor_deg_in: float = 60.0,
    sensor_deg_out: float = 300.0,
    mouth_deg_in: float = 40.0,
    mouth_deg_out: float = 340.0,
    eaten_distance: float = 25.0,
) -> None:
    agent_state, stepdf = load(logd, n_states)
    avgd_df = check_eaten(
        agent_state,
        stepdf,
        sensor_deg_in,
        sensor_deg_out,
        mouth_deg_in,
        mouth_deg_out,
        eaten_distance,
    )
    avgd_df.write_parquet(logd / "eaten.parquet")


if __name__ == "__main__":
    typer.run(main)
