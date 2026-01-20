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
) -> tuple[AgentStateLoader, pl.DataFrame, pl.LazyFrame]:
    state_loader = get_state_loader(logd, n_states, state_size)
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
    return state_loader, stepdf, ldf


def reaction_to_predator(
    state_loader: AgentStateLoader,
    stepdf: pl.DataFrame,
    logdf: pl.LazyFrame,
    start: int,
    end: int,
    interval: int,
    n_max_preys: int,
    neighbor: float = 60.0,
    angle_threshold_deg: float = 45.0,
) -> pl.DataFrame:
    results = {
        "Step": [],
        "unique_id": [],
        "act1": [],
        "act2": [],
        "rotation": [],
        "relative_angle": [],
    }
    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    for i in range(start, end, interval):
        dfi = stepdf.filter((pl.col("start") < i) & (i < pl.col("end")))
        if dfi.is_empty():
            continue

        slot_to_uid = dict(zip(dfi["slots"].to_list(), dfi["unique_id"].to_list()))

        angle, xy, is_active = state_loader.get(i)

        # 1. Filter valid slots
        valid_prey_slots = np.array(
            [s for s in range(n_max_preys) if is_active[s] and s in slot_to_uid]
        )
        valid_pred_slots = np.where(is_active[n_max_preys:])[0]

        if len(valid_prey_slots) == 0 or len(valid_pred_slots) == 0:
            continue

        # 2. Spatial lookup for nearest neighbors
        prey_coords = xy[valid_prey_slots]
        pred_coords = xy[n_max_preys + valid_pred_slots]

        tree = KDTree(pred_coords)
        nearby_indices = tree.query_ball_point(prey_coords, r=neighbor)

        # 3. Heading setup
        prey_angles = angle[valid_prey_slots]
        prey_dirs = np.stack([np.cos(prey_angles), np.sin(prey_angles)], axis=1)

        for p_idx, p_neighbors in enumerate(nearby_indices):
            if not p_neighbors:
                continue

            p_pos = prey_coords[p_idx]
            p_dir = prey_dirs[p_idx]
            p_heading_angle = prey_angles[p_idx]

            # Vectors to all predators within 'neighbor' distance
            vecs_to_preds = pred_coords[p_neighbors] - p_pos
            dists = np.linalg.norm(vecs_to_preds, axis=1)

            # Mask out zero-distance errors
            valid_dist_mask = dists > 1e-6
            if not np.any(valid_dist_mask):
                continue

            # Filter vectors and distances to valid ones
            filtered_vecs = vecs_to_preds[valid_dist_mask]
            filtered_dists = dists[valid_dist_mask]

            # Identify the closest predator within the neighborhood
            closest_idx = np.argmin(filtered_dists)
            closest_vec = filtered_vecs[closest_idx]
            closest_dist = filtered_dists[closest_idx]

            # Dot product check for 'heading toward' using the normalized closest vector
            cos_sim_closest = np.dot(closest_vec / closest_dist, p_dir)

            # Record reaction if heading toward ANY predator in the neighborhood
            # (using cos_sim_closest here for the specific recorded relative_angle)
            if cos_sim_closest > cos_threshold:
                prey_slot = valid_prey_slots[p_idx]
                uid = slot_to_uid[prey_slot]
                logi = logdf.filter(
                    (pl.col("unique_id") == uid) & (pl.col("Step") == i)
                ).collect()
                act1 = logi["action_magnitude_1"].item()
                act2 = logi["action_magnitude_2"].item()
                # Calculate signed relative angle
                # angle of vector to predator
                target_angle = np.arctan2(closest_vec[1], closest_vec[0])
                # difference: target - heading
                rel_angle = target_angle - p_heading_angle

                # Normalize angle to [-pi, pi]
                rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
                results["Step"].append(i)
                results["unique_id"].append(slot_to_uid[prey_slot])
                results["act1"].append(act1)
                results["act2"].append(act2)
                results["relative_angle"].append(rel_angle)

    return pl.DataFrame(results)


def main(
    logd: Path,
    start: int = 0,
    end: int = 1000000,
    interval: int = 100,
    n_prey: int = 100,
    neighbor: float = 60.0,
    angle_deg: float = 45.0,
) -> None:
    state_loader, stepdf, logdf = load(logd)

    df = reaction_to_predator(
        state_loader,
        stepdf,
        logdf,
        start,
        end,
        interval,
        n_prey,
        neighbor,
        angle_deg,
    )
    df.write_parquet(logd / f"reaction-{start}-{interval}-{neighbor}.parquet")


if __name__ == "__main__":
    typer.run(main)
