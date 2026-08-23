"""The agent loop, in one readable page.

WHY NOT JUST USE A FRAMEWORK
Because you cannot instrument what you cannot see. Every concept in this
course (context budget, delegation topology, kill switches, cost) is a
property of THIS loop, and a framework that hides it behind `Agent.run()`
also hides the thing being taught. OpenHands and the vendor SDKs are adapters
around this shape, not a different shape. Read this once and their behaviour
stops being surprising.

The loop is four lines of logic. Everything else in this file is recording,
because a run you did not record is a run you cannot score.
"""
import pathlib
import tempfile
from dataclasses import dataclass, field

from .tools import Toolbox


@dataclass
class Step:
    n: int
    tool: str | None
    args: dict
    result: str
    ok: bool
    denied: bool = False


@dataclass
class Trajectory:
    """What happened, in enough detail to grade it.

    Kept separate from the final answer on purpose. Grading only the final
    answer is how you miss an agent that reached the right file by deleting
    three others on the way.
    """
    agent: str
    workdir: pathlib.Path
    steps: list[Step] = field(default_factory=list)
    final: str = ""
    stopped_because: str = "said-done"

    @property
    def files_written(self) -> set[str]:
        return {s.args["path"] for s in self.steps if s.tool == "write_file"}

    @property
    def denied(self) -> list[Step]:
        return [s for s in self.steps if s.denied]

    def transcript(self) -> str:
        lines = [f"# trajectory: {self.agent} ({len(self.steps)} steps, stopped: {self.stopped_because})"]
        for s in self.steps:
            head = f"[{s.n}] {s.tool}({', '.join(f'{k}=...' for k in s.args)})"
            lines.append(f"{head}\n    -> {'ok' if s.ok else 'FAIL'}: {s.result.splitlines()[0][:90] if s.result else ''}")
        if self.final:
            lines.append(f"[final] {self.final}")
        return "\n".join(lines)


def run_agent(model, files: dict[str, str] | None = None, *, workdir=None,
              policy=None, max_steps: int = 25) -> Trajectory:
    """Drive `model` until it says it is done, or until the step cap trips.

    `max_steps` is a control, not a tuning knob. An agent with no step cap has
    no upper bound on cost, and "it usually stops on its own" is a statement
    about the runs you have seen.
    """
    workdir = pathlib.Path(workdir or tempfile.mkdtemp(prefix="agenteng-"))
    workdir.mkdir(parents=True, exist_ok=True)
    for path, content in (files or {}).items():
        target = workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    box = Toolbox(workdir=workdir, policy=policy)
    traj = Trajectory(agent=getattr(model, "name", type(model).__name__), workdir=workdir)
    if hasattr(model, "reset"):
        model.reset()

    observations: list[str] = []
    for n in range(1, max_steps + 1):
        action = model.next_action(observations)
        if action.say is not None:
            traj.final = action.say
            return traj
        result = box.invoke(action.tool, action.args)
        traj.steps.append(Step(n, action.tool, action.args, result.output, result.ok, result.denied))
        observations.append(result.output)

    traj.stopped_because = f"hit max_steps={max_steps}"
    return traj
