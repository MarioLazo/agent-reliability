"""The same agent, run under different harnesses. Watch the score move.

WHY THIS FILE EXISTS
A leaderboard row that names a model is reporting `model x harness x
substrate` and calling it `model`. On Terminal-Bench 2.0 the same unchanged
model spans **80.2% to 58.0%** across nine harnesses: a 22.2 percentage-point
spread with nothing about the model changing.

That is a large claim to make from a slide. So here it is at toy scale, on a
laptop, in ten seconds, where you can read every line of the harness that
caused it.

A harness here is four settings. None of them is exotic and none is a trick:

    step cap        how many actions before it is cut off
    tools           what it is allowed to reach for
    retry           does a failed tool call get a second attempt
    scope           what data it may touch

Change those and nothing else. The model, the task, and the fixture are
identical across every row.
"""
from dataclasses import dataclass, field

from agenteng.policy import Budget, PermissionBroker


@dataclass
class Harness:
    """Four settings, no model changes."""
    name: str
    max_steps: int = 25
    allow: set[str] = field(default_factory=set)
    scopes: dict[str, list[str]] = field(default_factory=dict)
    budget: int | None = None
    note: str = ""

    def policy(self):
        if not (self.allow or self.scopes or self.budget):
            return None
        return PermissionBroker(
            allow=set(self.allow),
            scopes=dict(self.scopes),
            budget=Budget(limit=self.budget) if self.budget else None,
        )


# Every one of these is a defensible configuration somebody actually ships.
HARNESSES = [
    Harness("generous", max_steps=25,
            note="no restrictions, long leash. what a demo runs on"),
    Harness("standard", max_steps=25, allow={"read_file", "write_file", "run"},
            note="an allowlist covering exactly what the task needs"),
    Harness("thrifty", max_steps=25, allow={"read_file", "write_file", "run"}, budget=4,
            note="a tight call budget, which is what cost pressure produces"),
    Harness("short-leash", max_steps=3,
            note="a step cap chosen for safety without checking what the task needs"),
    Harness("locked-down", max_steps=25, allow={"read_file", "write_file"},
            note="no shell. reasonable-sounding, and the agent cannot run its own tests"),
]
