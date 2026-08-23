#!/usr/bin/env python3
"""Build .ipynb files from percent-format .py sources.

WHY NOT EDIT THE NOTEBOOK DIRECTLY
A .ipynb is JSON carrying source, outputs, and execution counts in one blob.
Committed directly it produces diffs nobody reviews, merge conflicts nobody
can resolve, and a review culture where notebook changes are waved through.
For a course whose subject is review discipline, that is not a small
inconsistency.

So the .py source is what gets reviewed and the .ipynb is a build artifact.
`jupytext` does this and more; this is 60 lines of stdlib so a reader in
Colab installs nothing.

    python3 tools/nbbuild.py            # build all
    python3 tools/nbbuild.py --check    # fail if any .ipynb is stale (CI)
"""
import json
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "src"
OUT = SRC.parent


def parse(text: str) -> list[dict]:
    cells, kind, buf = [], "code", []

    def flush():
        source = "\n".join(buf).strip("\n")
        if not source:
            return
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": source.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": source.splitlines(keepends=True)})

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
            continue
        if kind == "markdown":
            # markdown cells are written as comments so the source file stays
            # valid, runnable Python. Strip exactly one leading "# ".
            buf.append(line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line))
        else:
            buf.append(line)
    flush()
    return cells


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for src in sorted(SRC.glob("*.py")):
        built = json.dumps(notebook(parse(src.read_text())), indent=1) + "\n"
        dest = OUT / (src.stem + ".ipynb")
        if check:
            if not dest.exists() or dest.read_text() != built:
                stale.append(dest.name)
        else:
            dest.write_text(built)
            print(f"built {dest.relative_to(OUT.parent)} ({len(parse(src.read_text()))} cells)")
    if stale:
        print(f"STALE: {', '.join(stale)}. Run `make notebooks`.", file=sys.stderr)
        return 1
    if check:
        print("notebooks up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
