"""A benchmark task, defined so that the three failures can separate.

THE SHAPE, AND WHY IT HAS FOUR TEST SLOTS INSTEAD OF ONE

    prompt          what the agent is told
    intent          what was actually needed, in English
    held_out_tests  written by the task author, never shown to the agent
    intent_probes   assertions about `intent`, not about `prompt`
    baseline        the boring version. no agent, no model, just code

Most benchmarks carry one test suite and report one number. That number
cannot distinguish "the agent did the wrong thing" from "I asked for the
wrong thing", and those need different fixes: one is a prompting problem,
the other is a specification problem. Keeping the probes separate from the
tests is what makes the Meaning Gap measurable instead of anecdotal.

`held_out` is load-bearing. Tests the agent can read are tests the agent can
satisfy without solving the problem, and that is not a hypothetical: it is the
single most common way an agent benchmark inflates.

`baseline` is the question almost nobody asks. **Only 39% of surveyed
production agent systems compare against a non-agentic baseline, and 26%
report that no meaningful baseline exists** (arXiv:2512.04123, 306
practitioners). Without one, "the agent scored 0.9" is unanchored: you cannot
tell a hard problem solved well from an easy problem solved expensively.

So every task here carries the boring version, and the harness scores it too.
Sometimes the honest finding is that a script would have done it.
"""
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    prompt: str
    intent: str
    files: dict[str, str] = field(default_factory=dict)
    held_out_tests: dict[str, str] = field(default_factory=dict)
    intent_probes: dict[str, str] = field(default_factory=dict)
    allowed_files: list[str] = field(default_factory=list)
    baseline: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if not self.held_out_tests:
            raise ValueError(f"task {self.id}: held_out_tests is required. "
                             "A task with no independent test is a demo.")
        if not self.intent_probes:
            raise ValueError(f"task {self.id}: intent_probes is required. "
                             "Without them you can only measure whether the agent "
                             "obeyed you, never whether obeying you was enough.")

    def in_scope(self, path: str) -> bool:
        """Was this file inside the boundary the task set?

        No globbing yet, deliberately. The moment scope becomes a pattern
        language, scope creep becomes arguable, and an arguable boundary is
        not a boundary.
        """
        return path in self.allowed_files
