from pathlib import Path

import polars as pl
import seaborn as sns
import typer

from emevo.environments.circle_foraging_with_predator import PREDATOR_COLOR


def main(dfile: Path) -> None:
    palette = sns.color_palette()
    df = pl.read_parquet(dfile)
    column_name = None
    for candidate in df.columns:
        if candidate.startswith("Descendant of"):
            column_name = candidate
            break

    if column_name is None:
        raise RuntimeError("No ancestor column found")

    RGB = "RGB"

    if "kind" in df.columns:

        def color_logic(index: int):
            return (
                pl.when(pl.col("kind") == 0)
                .then(
                    pl.when(pl.col(column_name))
                    .then(palette[1][index] * 255)
                    .otherwise(palette[0][index] * 255)
                    .cast(pl.Int32)
                )
                .otherwise(PREDATOR_COLOR[index])
                .alias(RGB[index])
            )
    else:

        def color_logic(index: int):
            return (
                pl.when(pl.col(column_name))
                .then(palette[1][index] * 255)
                .otherwise(palette[0][index] * 255)
                .alias(RGB[index])
                .cast(pl.Int32)
            )

    df = df.with_columns(
        color_logic(0),
        color_logic(1),
        color_logic(2),
        pl.lit(255).alias("A"),
    )
    df = df.select("unique_id", "R", "G", "B", "A")
    df.write_parquet(dfile.parent / (dfile.stem + "-rgba.parquet"))


if __name__ == "__main__":
    typer.run(main)
