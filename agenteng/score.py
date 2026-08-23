"""Three questions, scored separately, because they fail separately.

    correctness   did it do what I asked?      -> held-out tests
    quality       did it do it well?           -> scope, dependencies, size
    meaning       did I ask for the right thing? -> intent probes

Two more numbers are reported alongside them and neither is a score:

    self_consistency  do the agent's OWN tests pass?
    baseline          would a plain script have done this?

The first exists to be contrasted with correctness. When self_consistency is
green and correctness is red, you are looking at a closed loop: the agent
wrote an implementation and wrote the tests that agree with it, and both
encode the same misunderstanding. The pair of numbers is the finding. Either
number alone is misleading, which is why almost every agent demo shows you
exactly one of them.

The second exists to stop a score being unanchored. **An agent that passes a
task a five-line script also passes has not been shown to be worth anything**,
and only 39% of surveyed production systems check. Reporting it turns "the
agent scored 0.9" into "the agent scored 0.9 and the script scored 0.9", which
is a different conversation and usually a shorter one.
"""
import pathlib
import sys
from dataclasses import dataclass, field

from .tasks import Task
from .tools import imports_of, run_tests


@dataclass
class Verdict:
    passed: bool
    detail: str = ""
    ran: bool = True

    def __str__(self) -> str:
        if not self.ran:
            return "n/a"
        return "PASS" if self.passed else "FAIL"


@dataclass
class Quality:
    out_of_scope: list[str] = field(default_factory=list)
    new_imports: set[str] = field(default_factory=set)
    lines_changed: int = 0
    denied_calls: int = 0

    @property
    def clean(self) -> bool:
        return not self.out_of_scope and not self.new_imports

    def __str__(self) -> str:
        if self.clean:
            return f"clean ({self.lines_changed:+d} lines)"
        bits = []
        if self.out_of_scope:
            bits.append(f"scope creep: {', '.join(sorted(self.out_of_scope))}")
        if self.new_imports:
            bits.append(f"new deps: {', '.join(sorted(self.new_imports))}")
        return "; ".join(bits)


@dataclass
class Baseline:
    """How the boring version did on the same held-out tests and probes."""
    ran: bool = False
    correctness: bool = False
    meaning: bool = False

    def __str__(self) -> str:
        if not self.ran:
            return "n/a"
        return f"{'PASS' if self.correctness else 'FAIL'}/{'PASS' if self.meaning else 'FAIL'}"


@dataclass
class Score:
    task: str
    agent: str
    correctness: Verdict
    self_consistency: Verdict
    quality: Quality
    meaning: Verdict
    baseline: Baseline = field(default_factory=Baseline)

    @property
    def beat_the_baseline(self) -> str | None:
        """The question the field mostly does not ask.

        Returns None when there is no baseline to compare against, which is
        itself worth seeing in a report.
        """
        if not self.baseline.ran:
            return None
        # Compared dimension by dimension, not on a composite. The first
        # version collapsed both to a single pass/fail and reported "neither
        # works" for an agent that was exactly equivalent to a six-line
        # script, which is the one result the column exists to surface.
        agent = (self.correctness.passed, self.meaning.passed)
        base = (self.baseline.correctness, self.baseline.meaning)
        if agent == base:
            return "no lift over a plain script"
        if all(a >= b for a, b in zip(agent, base)):
            return "agent earns its keep"
        if all(b >= a for a, b in zip(agent, base)):
            return "the script beat the agent"
        return "mixed: better on one dimension, worse on another"

    @property
    def closed_loop(self) -> bool:
        """The agent's tests agree with the agent, and both are wrong."""
        return (self.self_consistency.ran and self.self_consistency.passed
                and not self.correctness.passed)

    @property
    def meaning_gap(self) -> bool:
        """Did exactly what was asked. Was not what was needed."""
        return self.correctness.passed and not self.meaning.passed


def _agent_test_files(traj) -> list[str]:
    """Test files the agent wrote itself, by naming convention."""
    return sorted(p for p in traj.files_written
                  if pathlib.Path(p).name.startswith("test_") and p.endswith(".py"))


def _local_modules(workdir: pathlib.Path) -> set[str]:
    """Module names importable from the workspace itself."""
    names = set()
    for entry in workdir.iterdir():
        if entry.suffix == ".py":
            names.add(entry.stem)
        elif entry.is_dir() and not entry.name.startswith("."):
            names.add(entry.name)
    return names


def _install(workdir: pathlib.Path, files: dict[str, str]) -> list[str]:
    for path, content in files.items():
        target = workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return list(files)


def _run_suite(workdir: pathlib.Path, files: list[str]) -> Verdict:
    if not files:
        return Verdict(False, "no test file", ran=False)
    outputs = []
    for f in files:
        r = run_tests(workdir, f)
        outputs.append(f"{f}: {'ok' if r.ok else 'FAILED'}")
        if not r.ok:
            tail = [l for l in r.output.splitlines() if l.strip()][-6:]
            return Verdict(False, f"{f}\n" + "\n".join(tail))
    return Verdict(True, "; ".join(outputs))


def _score_baseline(task: Task, wd_parent: pathlib.Path) -> Baseline:
    """Run the boring version against the same held-out tests and probes.

    In its own directory, so it cannot inherit anything the agent left behind.
    A baseline scored in the agent's workspace is not a baseline.
    """
    if not task.baseline:
        return Baseline()
    import tempfile
    wd = pathlib.Path(tempfile.mkdtemp(prefix="baseline-", dir=wd_parent))
    _install(wd, task.files)
    _install(wd, task.baseline)
    held = _run_suite(wd, _install(wd, task.held_out_tests))
    probes = _run_suite(wd, _install(wd, task.intent_probes))
    return Baseline(ran=True, correctness=held.passed, meaning=probes.passed)


def score(task: Task, traj) -> Score:
    """Grade one trajectory against one task.

    Order matters. The agent's own tests run FIRST, against the workspace as
    the agent left it, before the held-out tests are written to disk. If the
    held-out tests land first, the agent's suite can import them, and the
    independence you are trying to measure is gone.
    """
    wd = pathlib.Path(traj.workdir)

    self_v = _run_suite(wd, _agent_test_files(traj))

    held = _install(wd, task.held_out_tests)
    correct_v = _run_suite(wd, held)

    probes = _install(wd, task.intent_probes)
    meaning_v = _run_suite(wd, probes)

    baseline = set()
    for src in task.files.values():
        baseline |= imports_of(src)
    written = set()
    delta = 0
    for path in traj.files_written:
        src = (wd / path).read_text() if (wd / path).exists() else ""
        written |= imports_of(src)
        delta += len(src.splitlines()) - len(task.files.get(path, "").splitlines())

    # A "new dependency" means a third-party package, not `import customers`.
    # The first version of this counted local modules and reported a new
    # dependency on the file the agent had just been asked to write, which is
    # the kind of number that gets a quality gate switched off in week two.
    new_deps = written - baseline - _local_modules(wd) - sys.stdlib_module_names

    q = Quality(
        out_of_scope=[p for p in sorted(traj.files_written) if not task.in_scope(p)],
        new_imports=new_deps,
        lines_changed=delta,
        denied_calls=len(traj.denied),
    )
    return Score(task.id, traj.agent, correct_v, self_v, q, meaning_v,
                 _score_baseline(task, wd.parent))


def table(scores: list[Score]) -> str:
    """Render a results table. Findings are flagged, not left to be noticed."""
    w = max([len(s.agent) for s in scores] + [8])
    show_base = any(s.baseline.ran for s in scores)
    head = f"{'agent'.ljust(w)} | correct | own tests | meaning | quality"
    rows = [head, "-" * len(head)]
    for s in scores:
        rows.append(f"{s.agent.ljust(w)} | {str(s.correctness):^7} | "
                    f"{str(s.self_consistency):^9} | {str(s.meaning):^7} | {s.quality}")
    if show_base:
        b = scores[0].baseline
        rows.append(f"{'(baseline)'.ljust(w)} | {('PASS' if b.correctness else 'FAIL'):^7} | "
                    f"{'n/a':^9} | {('PASS' if b.meaning else 'FAIL'):^7} | no agent, no model")
    flags = []
    for s in scores:
        if s.closed_loop:
            flags.append(f"  CLOSED LOOP  {s.agent}: its own tests pass, the held-out tests do not.")
        if s.meaning_gap:
            flags.append(f"  MEANING GAP  {s.agent}: did what was asked, not what was needed.")
        verdict = s.beat_the_baseline
        if verdict == "no lift over a plain script":
            flags.append(f"  NO LIFT      {s.agent}: scores exactly what a six-line script scores.")
        elif verdict == "the script beat the agent":
            flags.append(f"  NEGATIVE     {s.agent}: the plain script did better.")
    if not show_base:
        flags.append("  NO BASELINE  nothing to compare against, so every score above is unanchored.")
    if flags:
        rows += ["", *flags]
    return "\n".join(rows)
