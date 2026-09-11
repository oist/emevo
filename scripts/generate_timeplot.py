"""Generate a timeplot notebook for a log directory."""

import json
import re
from pathlib import Path
from typing import Annotated

import typer


def main(
    log_directory: Annotated[
        Path, typer.Argument(help="Directory containing the logs.")
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir", help="Output directory (default: current directory)."
        ),
    ] = Path("."),
) -> None:
    log_directory = log_directory.expanduser().resolve()
    template = Path(__file__).resolve().parent / "assets" / "timeplot-template.ipynb"
    notebook = json.loads(template.read_text(encoding="utf-8"))
    replacement = f"Path({str(log_directory)!r})"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        updated = re.sub(r"\blog_directory\b", lambda _: replacement, "".join(source))
        cell["source"] = (
            updated.splitlines(keepends=True) if isinstance(source, list) else updated
        )

    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{log_directory.name}-timeplot.ipynb"
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    typer.echo(output)


if __name__ == "__main__":
    typer.run(main)
