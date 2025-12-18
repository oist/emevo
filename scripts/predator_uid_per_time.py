from pathlib import Path

import polars as pl
import typer

from emevo.analysis.log_plotting import load_log

REWARD_TYPES = [
    "Action Reward ($w_{act}$)",
    "Food Reward ($w_{food}$)",
    "Predator Reward ($w_{predator}$)",
    "Prey Reward ($w_{prey}$)",
]
REWAARD_KEYS = ["action", "food_1", "predator_sensor", "prey_sensor"]


def main(
    logd: Path,
    interval: int = 100000,
    max_steps: int = 10240000,
    n_states: int = 10,
) -> None:
    time_list = []
    rew_list = []
    rewtype_list = []
    uid_list = []
    logdf = (
        load_log(logd, last_idx=n_states)
        .with_columns(pl.col("step").alias("Step"))
        .sort("Step")
    )
    rdf = pl.read_parquet(logd / "profile_and_rewards.parquet")
    time_points = [(i + 1) * interval for i in range(max_steps // interval)]
    for time in time_points:
        uids = logdf.filter(pl.col("Step") == time).collect()["unique_id"]
        df = rdf.filter(pl.col("unique_id").is_in(uids))
        for rewtype, rewkey in zip(REWARD_TYPES, REWAARD_KEYS):
            rlen = len(df[rewkey])
            rew_list.append(df[rewkey])
            rewtype_list += [rewtype] * rlen
            time_list += [time] * rlen
            uid_list.append(df["unique_id"])
    df = pl.from_dict(
        {
            "Reward Value": pl.concat(rew_list),
            "Reward Type": rewtype_list,
            "unique_id": pl.concat(uid_list),
            "Step": time_list,
        }
    )
    df.write_parquet(logd / "uid_per_time.parquet")


if __name__ == "__main__":
    typer.run(main)
