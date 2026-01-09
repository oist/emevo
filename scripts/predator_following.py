import dataclasses
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import typer
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from emevo.analysis.log_plotting import load_log


class AgentState(NamedTuple):
    angle: NDArray
    xy: NDArray
    is_active: NDArray


@dataclasses.dataclass
class AgentStateLoader:
    size: int
    files: list[NpzFile]
    cache: dict[int, AgentState] = dataclasses.field(default_factory=dict)

    def get(self, time: int) -> AgentState:
        index = time // self.size
        if index not in self.cache:
            self._load_cache(index)
        cached = self.cache[index]
        offset = time % self.size
        angle = cached.angle[offset]
        xy = cached.xy[offset]
        is_active = cached.is_active[offset]
        return AgentState(angle=angle, xy=xy, is_active=is_active)

    def _load_cache(self, index: int) -> None:
        angle = self.files[index]["circle_axy"].astype(np.float32)[:, :, 0]
        xy = self.files[index]["circle_axy"].astype(np.float32)[:, :, 1:]
        is_active = self.files[index]["circle_is_active"].astype(bool)
        self.cache[index] = AgentState(angle=angle, xy=xy, is_active=is_active)


def get_state_loader(
    dirpath: Path,
    n_states: int,
    size: int = 1024000,
) -> AgentStateLoader:
    files = []
    for i in range(n_states):
        npzfile = np.load(dirpath / f"state-{i + 1}.npz")
        files.append(npzfile)
    return AgentStateLoader(size=size, files=files)


def load(
    logd: Path,
    n_states: int = 10,
    state_size: int = 1024000,
) -> tuple[AgentStateLoader, pl.DataFrame]:
    state_loader = get_state_loader(logd, n_states, state_size)

    eaten_path = logd / "eaten.parquet"
    if eaten_path.exists():
        stepdf = pl.read_parquet(eaten_path).select(
            "unique_id",
            "slots",
            "start",
            "end",
        )
    else:
        ldf = load_log(logd, last_idx=n_states).with_columns(
            pl.col("step").alias("Step")
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
    return state_loader, stepdf


def find_following_prey(
    *,
    state_loader: AgentStateLoader,
    stepdf: pl.DataFrame,
    start: int,
    interval: int,
    n_max_preys: int,
    end: int,
    neighbor: float = 50.0,
    angle_threshold: float = np.pi / 4,  # 45 degrees
) -> pl.DataFrame:
    step_list = []
    prey_uid_list = []
    predator_slot_list = []

    for i in range(start, end, interval):
        dfi = stepdf.filter((pl.col("start") < i) & (i < pl.col("end")))
        angle, xy, is_active = state_loader.get(i)

        # Split prey and predators
        all_prey_xy = xy[:n_max_preys]
        all_prey_active = is_active[:n_max_preys]
        all_prey_angles = angle[:n_max_preys]

        all_predator_xy = xy[n_max_preys:]
        all_predator_active = is_active[n_max_preys:]

        active_slots_in_df = set(dfi["slots"].to_list())

        # 1. Identify valid prey indices
        valid_prey_indices = [
            idx
            for idx in range(n_max_preys)
            if all_prey_active[idx] and idx in active_slots_in_df
        ]

        # 2. Identify valid predator indices (using offset for global indexing)
        valid_pred_indices = [
            idx for idx, active in enumerate(all_predator_active) if active
        ]

        if not valid_prey_indices or not valid_pred_indices:
            continue

        # 3. Compute Distance Matrix (Prey x Predators)
        prey_coords = all_prey_xy[valid_prey_indices]
        pred_coords = all_predator_xy[valid_pred_indices]
        dist_mat = cdist(prey_coords, pred_coords)

        # 4. Check Following Condition
        for p_idx, prey_slot in enumerate(valid_prey_indices):
            prey_pos = prey_coords[p_idx]
            prey_angle = all_prey_angles[prey_slot]

            # Unit vector of prey heading
            prey_dir = np.array([np.cos(prey_angle), np.sin(prey_angle)])

            for target_idx, pred_slot_local in enumerate(valid_pred_indices):
                dist = dist_mat[p_idx, target_idx]

                if dist < neighbor:
                    # Vector from prey to predator
                    vec_to_pred = pred_coords[target_idx] - prey_pos
                    vec_to_pred_unit = vec_to_pred / (
                        np.linalg.norm(vec_to_pred) + 1e-6
                    )

                    # Dot product to find cosine of angle between heading and predator
                    cos_sim = np.dot(prey_dir, vec_to_pred_unit)

                    # If angle difference is within threshold (e.g., cos(45°))
                    if cos_sim > np.cos(angle_threshold):
                        u_id = dfi.filter(pl.col("slots") == prey_slot)["unique_id"][0]

                        step_list.append(i)
                        prey_uid_list.append(u_id)
                        # predator index relative to the start of predator block
                        predator_slot_list.append(n_max_preys + pred_slot_local)

    return pl.DataFrame(
        {
            "Step": step_list,
            "prey_unique_id": prey_uid_list,
            "followed_predator_slot": predator_slot_list,
        }
    )


def main(
    logd: Path,
    n_states: int = 10,
    n_max_preys: int = 450,
    start: int = 9216000,
    interval: int = 1000,
    end: int = 10240000,
    neighbor: int = 25,
    state_size: int = 1024000,
) -> None:
    state_loader, stepdf = load(logd, n_states, state_size)
    group_df = find_following_prey(
        state_loader=state_loader,
        stepdf=stepdf,
        start=start,
        interval=interval,
        n_max_preys=n_max_preys,
        neighbor=neighbor,
        end=end,
    )
    group_df.write_parquet(logd / f"group-{start}-{interval}-{neighbor}.parquet")


if __name__ == "__main__":
    typer.run(main)
