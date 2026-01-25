import dataclasses
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import typer
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray


class AgentState(NamedTuple):
    xy: NDArray
    offset: int

    def _get(self, start: int, end: int) -> NDArray:
        return self.xy[start - self.offset : end - self.offset]


@dataclasses.dataclass
class AgentStateLoader:
    size: int
    files: list[NpzFile]
    cache: AgentState
    current_start_index: int = 0
    current_end_index: int = 0

    def get(self, start: int, end: int) -> NDArray:
        start_index = start // self.size
        end_index = end // self.size
        if (
            self.current_start_index <= start_index
            and end_index <= self.current_end_index
        ):
            return self.cache._get(start, end)
        # If not, extend the cache
        else:
            self._extend_cache(start_index, end_index)
            return self.cache._get(start, end)

    def _extend_cache(self, start_index: int, end_index: int) -> None:
        xy_next = self.files[end_index]["circle_axy"].astype(np.float32)[:, :, 1:]
        self.current_end_index = end_index
        if self.current_start_index < start_index:
            # Drop
            diff = start_index - self.current_start_index
            new_start = diff * self.size
            xy = np.concatenate((self.cache.xy[new_start:], xy_next))
            self.current_start_index = start_index
            print("Extend and drop")
        else:
            xy = np.concatenate((self.cache.xy, xy_next), axis=0)
            print("Extend")
        self.cache = AgentState(
            xy=xy,
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
    return AgentStateLoader(size=size, files=files, cache=AgentState(xy=xy, offset=0))


def load(
    logd: Path,
    n_states: int = 10,
    state_size: int = 1024000,
) -> tuple[AgentStateLoader, pl.DataFrame]:
    state_loader = get_state_loader(logd, n_states, state_size)
    age_path = logd / "age.parquet"
    stepdf = pl.read_parquet(age_path).select(
        "unique_id",
        "slots",
        "start",
        "end",
    )
    return state_loader, stepdf


def compute_avg_moved(
    state_loader: AgentStateLoader,
    stepdf: pl.DataFrame,
    n_max_preys: int,
) -> pl.DataFrame:
    prey_uid_list = []
    prey_avgd_list = []
    predator_uid_list = []
    predator_avgd_list = []
    for uid, slot, start, end in stepdf.sort("end").iter_rows():
        if end - start < 4:
            continue
        xy = state_loader.get(start, end)  # (end - start, 500, 2)
        xy_selected = xy[:, slot]
        xy0 = xy_selected[:-1]
        xy1 = xy_selected[1:]
        avgd = np.mean(np.linalg.norm(xy0 - xy1, axis=1))
        assert not np.isnan(avgd), (start, end, slot)
        if slot < n_max_preys:
            prey_uid_list.append(uid)
            prey_avgd_list.append(avgd)
        else:
            predator_uid_list.append(uid)
            predator_avgd_list.append(avgd)

    species = ["Prey"] * len(prey_uid_list) + ["Predator"] * len(predator_uid_list)
    avgd_df = pl.from_dict(
        {
            "unique_id": prey_uid_list + predator_uid_list,
            "Avg. Distances": prey_avgd_list + predator_avgd_list,
            "Species": species,
        }
    )
    return avgd_df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(logd: Path, n_max_preys: int = 450) -> None:
    state_loader, stepdf = load(logd)
    avgd_df = compute_avg_moved(state_loader, stepdf, n_max_preys)
    avgd_df.write_parquet(logd / "avgd.parquet")


if __name__ == "__main__":
    typer.run(main)
