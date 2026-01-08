import dataclasses
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import typer
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from emevo.analysis.log_plotting import load_log


class AgentState(NamedTuple):
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
        xy = cached.xy[offset]
        is_active = cached.is_active[offset]
        return AgentState(xy=xy, is_active=is_active)

    def _load_cache(self, index: int) -> None:
        xy = self.files[index]["circle_axy"].astype(np.float32)[:, :, 1:]
        is_active = self.files[index]["circle_is_active"].astype(bool)
        self.cache[index] = AgentState(xy=xy, is_active=is_active)


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


def find_groups(
    state_loader: AgentStateLoader,
    stepdf: pl.DataFrame,
    start: int,
    interval: int,
    n_max_preys: int,
    end: int,
) -> pl.DataFrame:
    step_list = []
    uid_list = []
    group_list = []
    size_list = []
    x_list = []
    y_list = []

    for i in range(start, end, interval):
        # list up all agents that exists at this time
        dfi = stepdf.filter((pl.col("start") < i) & (i < pl.col("end")))
        xy, is_active = state_loader.get(i)
        # Complete the code here:
        all_prey_xy = xy[:n_max_preys]
        all_prey_active = is_active[:n_max_preys]

        # 1. Identify valid indices based on:
        # - Agent is marked active in the state loader
        # - Agent is far from walls (60 < x, y < 900)
        # - Agent exists in the dfi (has a unique_id record)
        active_slots_in_df = set(dfi["slots"].to_list())

        valid_indices = []
        for idx in range(n_max_preys):
            if all_prey_active[idx] and idx in active_slots_in_df:
                pos = all_prey_xy[idx]
                if (60 < pos[0] < 900) and (60 < pos[1] < 900):
                    valid_indices.append(idx)

        if len(valid_indices) < 1:
            continue

        # 2. Compute the distance matrix for valid agents
        valid_xy = all_prey_xy[valid_indices]
        # pdist returns condensed distance vector, squareform converts to NxN matrix
        dist_mat = squareform(pdist(valid_xy))

        # 3. Create an adjacency matrix where distance < 25
        # This acts as the connectivity requirement for our "groups"
        adj_matrix = dist_mat < 25

        # 4. Use connected components to label the groups (similar to Union-Find)
        _, labels = connected_components(
            csgraph=csr_matrix(adj_matrix),
            directed=False,
            return_labels=True,
        )

        unique_groups, counts = np.unique(labels, return_counts=True)
        group_to_size = dict(zip(unique_groups, counts))

        # 5. Map the results back to the requested lists
        for local_idx, group_id in enumerate(labels):
            original_slot = valid_indices[local_idx]
            xy = all_prey_xy[original_slot]
            # Extract unique_id from dfi for this specific slot
            u_id = dfi.filter(pl.col("slots") == original_slot)["unique_id"][0]

            step_list.append(i)
            uid_list.append(u_id)
            group_list.append(group_id)
            size_list.append(group_to_size[group_id])
            x_list.append(xy[0])
            y_list.append(xy[1])

    df = pl.from_dict(
        {
            "Step": step_list,
            "unique_id": uid_list,
            "Group": group_list,
            "Group Size": size_list,
            "x": x_list,
            "y": y_list,
        }
    )
    return df


def main(
    logd: Path,
    n_states: int = 10,
    n_max_preys: int = 450,
    start: int = 9216000,
    interval: int = 1000,
    end: int = 10240000,
    state_size: int = 1024000,
) -> None:
    state_loader, stepdf = load(logd, n_states, state_size)
    group_df = find_groups(state_loader, stepdf, start, interval, n_max_preys, end)
    group_df.write_parquet(logd / "group.parquet")
    # For debug
    # print(group_df.sort("group"))


if __name__ == "__main__":
    typer.run(main)
