from pathlib import Path

import polars as pl
import seaborn as sns
import typer


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

    df = df.with_columns(
        pl.when(pl.col(column_name))
        .then(palette[0][0] * 255)
        .otherwise(palette[1][0] * 255)
        .alias("R")
        .cast(pl.Int32),
        pl.when(pl.col(column_name))
        .then(palette[0][1] * 255)
        .otherwise(palette[1][1] * 255)
        .alias("G")
        .cast(pl.Int32),
        pl.when(pl.col(column_name))
        .then(palette[0][2] * 255)
        .otherwise(palette[1][2] * 255)
        .alias("B")
        .cast(pl.Int32),
        pl.lit(255).alias("A"),
    )
    df = df.select("unique_id", "R", "G", "B", "A")
    df.write_parquet(dfile.parent / (dfile.stem + "-rgba.parquet"))


if __name__ == "__main__":
    typer.run(main)
