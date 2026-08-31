"""Remove transient Jupyter state from notebooks before committing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def clean_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Remove execution artifacts while retaining notebook source and metadata."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

        metadata = cell.get("metadata", {})
        for key in ("ExecuteTime", "collapsed", "execution", "scrolled", "trusted"):
            metadata.pop(key, None)

    notebook.get("metadata", {}).pop("widgets", None)
    return notebook


def serialize(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()


def clean_staged_notebooks() -> int:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            "--",
            "*.ipynb",
        ],
        check=True,
        capture_output=True,
    )
    paths = [path for path in result.stdout.split(b"\0") if path]
    for raw_path in paths:
        path = raw_path.decode()
        staged = subprocess.run(
            ["git", "show", f":{path}"], check=True, capture_output=True
        ).stdout
        cleaned = serialize(clean_notebook(json.loads(staged)))
        if cleaned == staged:
            continue

        object_id = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            input=cleaned,
            check=True,
            capture_output=True,
            text=False,
        ).stdout.decode().strip()
        mode = subprocess.run(
            ["git", "ls-files", "-s", "--", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split(maxsplit=1)[0]
        subprocess.run(
            ["git", "update-index", "--cacheinfo", mode, object_id, path], check=True
        )
        print(f"Cleaned staged notebook: {path}")
    return len(paths)


def clean_files(paths: list[Path]) -> int:
    for path in paths:
        notebook = json.loads(path.read_text())
        path.write_bytes(serialize(clean_notebook(notebook)))
        print(f"Cleaned notebook: {path}")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--staged", action="store_true", help="clean notebooks in the Git index"
    )
    args = parser.parse_args()
    if args.staged:
        if args.paths:
            parser.error("paths cannot be combined with --staged")
        clean_staged_notebooks()
    elif args.paths:
        clean_files(args.paths)
    else:
        parser.error("provide notebook paths or --staged")


if __name__ == "__main__":
    main()
