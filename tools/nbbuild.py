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
    python3 tools/nbbuild.py --check    # fail if any .ipynb is stale, or any
                                        # footer disagrees with ORDER (CI)
"""
import json
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "src"
OUT = SRC.parent

# The one source of truth for the series name and the notebook order.
# Footers are generated from this rather than typed, because typed footers
# drifted: 01 advertised a "Context Engineering" notebook that does not exist,
# 02 named the wrong series, and 05 claimed to be in Part 3 of a Part 2 repo.
SERIES = "Agent Reliability Engineering"
ORDER = [
    ("01_evaluation", "01 · Evaluating Agent Work"),
    ("02_judging_the_judge", "02 · Judging the Judge"),
    ("03_delegation", "03 · Delegation"),
    ("04_guardrails", "04 · Guardrails"),
    ("05_voice", "05 · Voice"),
    ("06_benchmarking", "06 · Benchmarks and the Oracle Problem"),
]


def footer_for(stem: str) -> str:
    """The footer line this notebook must end with."""
    stems = [s for s, _ in ORDER]
    if stem not in stems:
        raise SystemExit(f"{stem} is not in ORDER; add it to tools/nbbuild.py")
    i = stems.index(stem)
    parts = [f"Part of *{SERIES}*."]
    if i > 0:
        parts.append(f"Previous: **{ORDER[i - 1][1]}**.")
    if i < len(ORDER) - 1:
        parts.append(f"Next: **{ORDER[i + 1][1]}**.")
    return "# " + " ".join(parts)


def source_problems(name: str, text: str) -> list[str]:
    """Every SOURCES entry must carry a non-empty citation, and a file that
    declares SOURCES must print them, or the provenance never reaches a reader.

    This exists because figures the harness does not produce were sitting bare
    in the prose under a footer claiming the harness produced everything.
    """
    if "SOURCES = {" not in text:
        return []
    problems = []
    body = text.split("SOURCES = {", 1)[1].split("}", 1)[0]
    for entry in re.findall(r'"([^"]+)":\s*\n?\s*"([^"]*)"', body):
        figure, citation = entry
        if not citation.strip():
            problems.append(f"{name}: SOURCES['{figure}'] has no citation")
    if "for _figure, _source in SOURCES.items():" not in text:
        problems.append(f"{name}: declares SOURCES but never prints them")
    return problems


def footer_of(text: str) -> str:
    """The last comment line of a source file."""
    for line in reversed(text.strip().splitlines()):
        if line.startswith("#"):
            return line.rstrip()
    return ""


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
    bad_footers = []
    bad_sources = []
    for src in sorted(SRC.glob("*.py")):
        text = src.read_text()
        want = footer_for(src.stem)
        bad_sources.extend(source_problems(src.name, text))
        if footer_of(text) != want:
            bad_footers.append(f"{src.name}\n     want: {want}\n     got:  {footer_of(text)}")
        built = json.dumps(notebook(parse(text)), indent=1) + "\n"
        dest = OUT / (src.stem + ".ipynb")
        if check:
            if not dest.exists() or dest.read_text() != built:
                stale.append(dest.name)
        else:
            dest.write_text(built)
            print(f"built {dest.relative_to(OUT.parent)} ({len(parse(src.read_text()))} cells)")
    if bad_sources:
        print("UNCITED FIGURES:", file=sys.stderr)
        for b in bad_sources:
            print(f"  {b}", file=sys.stderr)
        return 1
    if bad_footers:
        print("FOOTER DRIFT (ORDER in tools/nbbuild.py is the source of truth):",
              file=sys.stderr)
        for b in bad_footers:
            print(f"  {b}", file=sys.stderr)
        return 1
    if stale:
        print(f"STALE: {', '.join(stale)}. Run `make notebooks`.", file=sys.stderr)
        return 1
    if check:
        print("notebooks up to date, footers match ORDER, figures cited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
