import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray

from emevo.analysis.log_plotting import load_log


@dataclasses.dataclass
class AgentState:
    xy: NDArray
    is_active: NDArray


def load_agent_state(dirpath: Path, n_states: int) -> AgentState:
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


def load(logd: Path, n_states: int = 10) -> tuple[AgentState, pl.DataFrame]:
    ldf = load_log(logd, last_idx=n_states).with_columns(pl.col("step").alias("Step"))
    agent_state = load_agent_state(logd, n_states)
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


def mean_masked_norm(a: NDArray, b: NDArray, mask: NDArray) -> NDArray:
    norm = np.sum(np.square(a - b), axis=-1)  # (N, M)
    return np.mean(norm * mask)  # (1,)


def compute_dxy_dist(
    agent_state: AgentState,
    stepdf: pl.DataFrame,
    n_max_preys: int,
) -> pl.DataFrame:
    def dxy_dist(start: int, end: int, slot: int) -> tuple[NDArray, NDArray]:
        xy_selected = agent_state.xy[start:end, slot]
        prey_xy = agent_state.xy[start:end, :n_max_preys]
        predator_xy = agent_state.xy[start:end, n_max_preys:]
        # dxy
        dxy_self = xy_selected[1:] - xy_selected[:-1]  # (N - 1, 2)
        dxy_self_expanded = np.expand_dims(dxy_self, axis=1)
        dxy_prey = prey_xy[1:] - prey_xy[:-1]  # (N - 1, M1, 2)
        dxy_predator = predator_xy[1:] - predator_xy[:-1]  # (N - 1, M2, 2)
        # mask
        prey_mask = np.logical_and(
            agent_state.is_active[start + 1 : end, :n_max_preys],
            agent_state.is_active[start : end - 1, :n_max_preys],
        )
        predator_mask = np.logical_and(
            agent_state.is_active[start + 1 : end, n_max_preys:],
            agent_state.is_active[start : end - 1, n_max_preys:],
        )
        to_prey = mean_masked_norm(dxy_self_expanded, dxy_prey, prey_mask)
        to_predator = mean_masked_norm(dxy_self_expanded, dxy_predator, predator_mask)
        return to_prey, to_predator

    uid_list = []
    prey_list = []
    predator_list = []
    for uid, slot, start, end in stepdf.iter_rows():
        if slot >= n_max_preys:  # It's predator
            continue
        if end - start < 2:
            continue
        to_prey, to_predator = dxy_dist(start, end, slot)
        uid_list.append(uid)
        prey_list.append(to_prey)
        predator_list.append(to_predator)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Prey Dist.": prey_list,
            "Predator Dist.": predator_list,
        }
    )
    return df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(logd: Path, n_states: int = 10, n_max_preys: int = 450) -> None:
    agent_state, stepdf = load(logd, n_states)
    avgd_df = compute_dxy_dist(agent_state, stepdf, n_max_preys)
    avgd_df.write_parquet(logd / "avg-movement-sim.parquet")


if __name__ == "__main__":
    typer.run(main)
