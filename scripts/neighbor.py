import dataclasses
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import typer
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from emevo.analysis.log_plotting import load_log


class AgentState(NamedTuple):
    xy: NDArray
    is_active: NDArray
    offset: int


@dataclasses.dataclass
class AgentStateLoader:
    size: int
    files: list[NpzFile]
    cache: AgentState
    current_start_index: int = 0
    current_end_index: int = 0

    def get(self, start: int, end: int) -> AgentState:
        start_index = start // self.size
        end_index = end // self.size
        if (
            self.current_start_index <= start_index
            and end_index <= self.current_end_index
        ):
            return self.cache
        # If not, extend the cache
        else:
            self._extend_cache(start_index, end_index)
            return self.cache

    def _extend_cache(self, start_index: int, end_index: int) -> None:
        xy_next = self.files[end_index]["circle_axy"].astype(np.float32)[:, :, 1:]
        is_active_next = self.files[end_index]["circle_is_active"].astype(bool)
        self.current_end_index = end_index
        if self.current_start_index < start_index:
            # Drop
            diff = start_index - self.current_start_index
            new_start = diff * self.size
            xy = np.concatenate((self.cache.xy[new_start:], xy_next))
            is_active = np.concatenate(
                (self.cache.is_active[new_start:], is_active_next)
            )
            self.current_start_index = start_index
            print("Extend and drop")
        else:
            xy = np.concatenate((self.cache.xy, xy_next), axis=0)
            is_active = np.concatenate((self.cache.is_active, is_active_next), axis=0)
            print("Extend")
        self.cache = AgentState(
            xy=xy,
            is_active=is_active,
            offset=start_index * self.size,
        )


def get_state_loader(
    dirpath: Path,
    n_states: int,
    size: int = 1024000,
) -> AgentStateLoader:
    files = []
    for i in range(n_states):
        npzfile = np.load(dirpath / f"state-{i + 1}.npz")
        files.append(npzfile)
    xy = files[0]["circle_axy"].astype(np.float32)[:, :, 1:]
    is_active = files[0]["circle_is_active"].astype(bool)
    return AgentStateLoader(
        size=size, files=files, cache=AgentState(xy=xy, is_active=is_active, offset=0)
    )


def load(
    logd: Path,
    n_states: int = 10,
    state_size: int = 1024000,
) -> tuple[AgentStateLoader, pl.DataFrame]:
    state_loader = get_state_loader(logd, n_states, state_size)

    age_path = logd / "age.parquet"
    if age_path.exists():
        stepdf = pl.read_parquet(age_path).select(
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


def mean_masked_norm(a: NDArray, b: NDArray, mask: NDArray):
    norm = np.sum(np.square(a - b), axis=-1)  # (N, M)
    return np.sum(norm * mask) / np.sum(mask)


def compute_n_neighbors(
    state_loader: AgentStateLoader,
    stepdf: pl.DataFrame,
    n_max_preys: int,
    threshold: float,
    n_max_iter: int | None = None,  # For debugging
) -> pl.DataFrame:
    def n_neighbors(start: int, end: int, slot: int) -> NDArray:
        xy, is_active, offset = state_loader.get(start, end)
        xy_selected = xy[start - offset : end - offset + 1, slot]  # (N, 2)
        prey_xy = xy[start - offset : end - offset + 1, :n_max_preys]  # (N, M, 2)
        prey_is_active = is_active[start - offset : end - offset + 1, :n_max_preys]
        distance = np.linalg.norm(
            prey_xy - np.expand_dims(xy_selected, axis=1),
            axis=-1,
        )
        is_neighbor = (distance < threshold) & prey_is_active
        return np.mean(np.sum(is_neighbor, axis=1))

    uid_list = []
    neighbor_list = []
    for i, (uid, slot, start, end) in enumerate(stepdf.sort("start").iter_rows()):
        if n_max_iter is not None and n_max_iter < i:
            break
        if end - start < 2:
            continue
        if slot >= n_max_preys:
            continue
        n_avg_neighbors = n_neighbors(start, end, slot)
        uid_list.append(uid)
        neighbor_list.append(n_avg_neighbors)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "Num. Neighbors": neighbor_list,
        }
    )
    return df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(
    logd: Path,
    n_states: int = 10,
    n_max_preys: int = 450,
    threshold: float = 50.0,
    state_size: int = 1024000,
) -> None:
    state_loader, stepdf = load(logd, n_states, state_size)
    avgd_df = compute_n_neighbors(state_loader, stepdf, n_max_preys, threshold)
    avgd_df.write_parquet(logd / "neighbor.parquet")


if __name__ == "__main__":
    typer.run(main)
