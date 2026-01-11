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


@dataclasses.dataclass
class AgentStateLoader:
    size: int
    files: list[NpzFile]
    cache: AgentState
    current_index: int = 0

    def query(self, step: int) -> AgentState:
        index = step // self.size
        if index != self.current_index:
            self._swap_cache(index)
        offset = self.current_index * self.size
        return AgentState(
            xy=self.cache.xy[step - offset],
            is_active=self.cache.is_active[step - offset],
        )

    def _swap_cache(self, index: int) -> None:
        xy_next = self.files[index]["circle_axy"].astype(np.float32)[:, :, 1:]
        is_active_next = self.files[index]["circle_is_active"].astype(bool)
        self.current_index = index
        self.cache = AgentState(xy_next, is_active_next)


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
        size=size,
        files=files,
        cache=AgentState(xy=xy, is_active=is_active),
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


def check_death_place(loader: AgentStateLoader, stepdf: pl.DataFrame) -> pl.DataFrame:
    uid_list = []
    x_list = []
    y_list = []
    for uid, slot, start, end in stepdf.sort("end").iter_rows():
        if end - start < 2:
            continue
        uid_list.append(uid)
        state = loader.query(end)
        x, y = state.xy[slot]
        x_list.append(x)
        y_list.append(y)
    df = pl.from_dict(
        {
            "unique_id": uid_list,
            "x": x_list,
            "y": y_list,
        }
    )
    return df.join(
        stepdf.with_columns(pl.col("unique_id").cast(pl.Int64)),
        on="unique_id",
    )


def main(logd: Path, n_states: int = 10) -> None:
    agent_state, stepdf = load(logd, n_states)
    dp_df = check_death_place(agent_state, stepdf)
    dp_df.write_parquet(logd / "death_place.parquet")


if __name__ == "__main__":
    typer.run(main)
