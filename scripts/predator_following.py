import dataclasses
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import typer
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from scipy.spatial import KDTree

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
    angle_threshold_deg: float = 45,
) -> pl.DataFrame:
    step_list = []
    uid_list = []
    is_following_list = []
    cos_threshold = np.cos(np.radians(angle_threshold_deg))

    for i in range(start, end, interval):
        # 1. Fast ID Lookup
        dfi = stepdf.filter((pl.col("start") < i) & (i < pl.col("end")))
        if dfi.is_empty():
            continue

        angle, xy, is_active = state_loader.get(i)
        slot_to_uid = dict(zip(dfi["slots"].to_list(), dfi["unique_id"].to_list()))
        is_in_valid_range = ((60 < xy[:, 0]) & (xy[:, 0] < 900)) & (
            (60 < xy[:, 1]) & (xy[:, 1] < 900)
        )
        is_valid = is_active & is_in_valid_range

        # 2. Filter Active Agents
        valid_prey_slots = np.where(is_valid[:n_max_preys])[0]
        valid_pred_slots = np.where(is_valid[n_max_preys:])[0]

        if len(valid_prey_slots) == 0 or len(valid_pred_slots) == 0:
            continue

        # 3. Spatial Indexing with KDTree
        prey_coords = xy[valid_prey_slots]
        pred_coords = xy[n_max_preys + valid_pred_slots]

        # Build tree for predators, query with prey
        tree = KDTree(pred_coords)
        # Returns list of lists: for each prey, local indices of nearby predators
        nearby_indices = tree.query_ball_point(prey_coords, r=neighbor)

        # 4. Heading Alignment Check
        prey_angles = angle[valid_prey_slots]
        # Pre-calculate prey heading unit vectors
        prey_dirs = np.stack([np.cos(prey_angles), np.sin(prey_angles)], axis=1)

        for p_idx, p_neighbors in enumerate(nearby_indices):
            prey_slot = valid_prey_slots[p_idx]
            uid = slot_to_uid[prey_slot]
            step_list.append(i)
            uid_list.append(uid)
            if len(p_neighbors) == 0:
                is_following_list.append(False)
                continue

            p_pos = prey_coords[p_idx]
            p_dir = prey_dirs[p_idx]

            # Sub-selection of nearby predators
            targets_pos = pred_coords[p_neighbors]

            # Vector from prey to predators
            vecs_to_preds = targets_pos - p_pos
            dists = np.linalg.norm(vecs_to_preds, axis=1)

            # Avoid division by zero
            valid_mask = dists > 1e-6
            if not np.any(valid_mask):
                continue

            # Normalize vectors and calculate dot product
            unit_vecs = vecs_to_preds[valid_mask] / dists[valid_mask, np.newaxis]
            cos_sims = np.dot(unit_vecs, p_dir)

            # Check angle threshold
            is_following = cos_sims > cos_threshold
            is_following_list.append(bool(np.any(is_following)))

    return pl.DataFrame(
        {
            "Step": step_list,
            "unique_id": uid_list,
            "is_following": is_following_list,
        }
    )


def main(
    logd: Path,
    n_states: int = 10,
    n_max_preys: int = 450,
    start: int = 9216000,
    interval: int = 1000,
    end: int = 10240000,
    neighbor: int = 30,
    state_size: int = 1024000,
    angle_deg: float = 45.0,
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
        angle_threshold_deg=angle_deg,
    )
    group_df.write_parquet(logd / f"following-{start}-{interval}-{neighbor}.parquet")


if __name__ == "__main__":
    typer.run(main)
