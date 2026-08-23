"""Run every agent against every task. The reproducible table.

Deliberately not a pytest fixture: this is the artifact a student re-runs when
they change a model, adopt a skill, or rewrite a prompt. It has to be one
command with no arguments, or it will not get re-run, and a benchmark that is
not re-run is a screenshot.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agenteng.loop import run_agent
from agenteng.score import score, table
from bench.agents.dedupe_agents import ALL
from bench.tasks.dedupe import TASK

SUITE = [(TASK, ALL)]


def main() -> int:
    findings = 0
    for task, agents in SUITE:
        print(f"\n## {task.id}\n")
        scores = [score(task, run_agent(m, files=task.files)) for m in agents]
        print(table(scores))
        findings += sum(s.closed_loop or s.meaning_gap for s in scores)
    print(f"\n{findings} finding(s) across {len(SUITE)} task(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
