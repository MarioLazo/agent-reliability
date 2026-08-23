"""Meta-evaluation by corruption. Does your judge actually read the reference?

    Flip the labels on 10% of your golden set. If your judge's aggregate score
    does not move, it is not reading the answer.

That is the headline, and this module makes it runnable. It is the
software-testing move applied one level up: mutation testing breaks the code
and asks whether the tests notice; this breaks the reference and asks whether
the judge notices.

TWO CORRUPTIONS, AND THEY ARE NOT EQUIVALENT
--------------------------------------------
`flip` inverts the gold verdict on a sample of rows. `swap` keeps every label
and permutes which reference goes with which output.

They measure different things, and teaching them as one test is the mistake
worth avoiding:

  flip  penalises any judge whose verdicts are fixed, which includes a good
        one. Its discriminating power scales with the judge's baseline
        accuracy: against a coin-flip judge, expected movement is zero, so
        flip cannot distinguish a bad judge from a lucky one.

  swap  is the sharp instrument. A judge that reads THIS reference produces
        different verdicts once references are permuted, so its score falls.
        A judge that ignores the reference produces identical verdicts against
        identical labels, so its score moves by exactly zero.

Run both. Report both. `swap` is the one that answers the question in the
headline; `flip` is the one people already know.

WHAT THIS CANNOT TEST
---------------------
A reference-free judge, shown a rubric and never a gold answer, cannot fail
either test, because there is no reference to corrupt. Running it there
produces a reassuring zero that means nothing. `meta_evaluate` refuses a
verdict in that case rather than emitting the comfortable number: a
meta-evaluation that cannot fail is exactly the thing to be suspicious of.
"""
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

Judge = Callable[[Sequence[dict]], list[str]]
LABELS = ("pass", "fail")


def load(path) -> list[dict]:
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        missing = {"id", "output", "reference", "label"} - set(r)
        if missing:
            raise ValueError(f"row {i}: missing {sorted(missing)}")
        if r["label"] not in LABELS:
            raise ValueError(f"row {i}: label must be pass|fail, got {r['label']!r}")
    return rows


def corrupt(rows: Sequence[dict], mode: str, frac: float, seed: int) -> tuple[list[dict], list[int]]:
    """Return a corrupted copy and the indices touched. Seeded, so it repeats."""
    rng = random.Random(seed)
    n = max(1, round(len(rows) * frac))
    idx = sorted(rng.sample(range(len(rows)), n))
    out = [dict(r) for r in rows]
    if mode == "flip":
        for i in idx:
            out[i]["label"] = "fail" if out[i]["label"] == "pass" else "pass"
    elif mode == "swap":
        # A rotation of one element is the identity, so a swap that touches a
        # single row corrupts nothing and reports a clean pass. That is the
        # worst possible failure for a test whose job is to detect a judge
        # that notices nothing, so it is an error rather than a quiet no-op.
        if len(idx) < 2:
            raise ValueError(
                f"swap needs at least 2 rows to permute, got {len(idx)} "
                f"(frac={frac} of {len(rows)} rows). Raise frac or the set size."
            )
        refs = [out[i]["reference"] for i in idx]
        for i, r in zip(idx, refs[1:] + refs[:1]):
            out[i]["reference"] = r
    else:
        raise ValueError(f"unknown corruption {mode!r}, expected flip or swap")
    return out, idx


# --- judges ---------------------------------------------------------------
# Three, so the results table has a middle. None is a straw man.

def honest_judge(rows: Sequence[dict]) -> list[str]:
    """Reads the reference and compares to it."""
    return ["pass" if r["output"].strip().lower() == r["reference"].strip().lower()
            else "fail" for r in rows]


def blind_judge(rows: Sequence[dict]) -> list[str]:
    """Never looks at the reference. Passes anything that looks like an answer.

    Not a straw man: a judge prompted "is this a good response?" with no
    reference in its context behaves exactly like this, and that prompt is
    extremely common.
    """
    return ["pass" if len(r["output"].strip()) > 3 else "fail" for r in rows]


def overlap_judge(rows: Sequence[dict]) -> list[str]:
    """Reads the reference, but only for token overlap. The realistic middle.

    Survives naive cases and fails on near-misses, which is where the
    expensive errors live.
    """
    out = []
    for r in rows:
        a = set(r["output"].lower().split())
        b = set(r["reference"].lower().split())
        out.append("pass" if a and b and len(a & b) / len(a | b) >= 0.5 else "fail")
    return out


def agreement(verdicts: Sequence[str], rows: Sequence[dict]) -> float:
    return sum(v == r["label"] for v, r in zip(verdicts, rows)) / len(rows)


@dataclass
class MetaResult:
    judge: str
    mode: str
    clean: float
    corrupted: float
    expected: float
    touched: int
    total: int
    reference_free: bool = False

    @property
    def delta(self) -> float:
        return self.clean - self.corrupted

    @property
    def ratio(self) -> float:
        return self.delta / self.expected if self.expected else 0.0

    @property
    def verdict(self) -> str:
        if self.reference_free:
            return "N/A"
        if self.mode == "swap":
            return "DISCRIMINATING" if self.delta > 0.01 else "NOT READING THE REFERENCE"
        if self.ratio >= 0.7:
            return "DISCRIMINATING"
        if self.ratio >= 0.3:
            return "WEAK"
        return "NOT READING THE REFERENCE"


def meta_evaluate(rows: Sequence[dict], judge: Judge, *, mode: str = "flip",
                  frac: float = 0.10, seed: int = 20260823,
                  name: str | None = None, reference_free: bool = False) -> MetaResult:
    dirty, idx = corrupt(rows, mode, frac, seed)
    return MetaResult(
        judge=name or getattr(judge, "__name__", "judge"),
        mode=mode,
        clean=agreement(judge(rows), rows),
        corrupted=agreement(judge(dirty), dirty),
        expected=len(idx) / len(rows),
        touched=len(idx),
        total=len(rows),
        reference_free=reference_free,
    )


def table(results: Sequence[MetaResult]) -> str:
    w = max([len(r.judge) for r in results] + [5])
    head = f"{'judge'.ljust(w)} | mode | clean | corrupted |  delta | verdict"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(f"{r.judge.ljust(w)} | {r.mode:^4} | {r.clean:.3f} |   {r.corrupted:.3f}   | "
                     f"{r.delta:+.3f} | {r.verdict}")
    if any(r.reference_free for r in results):
        lines += ["", "  N/A: judge declared reference-free. There is no reference to corrupt,",
                  "       so this test cannot fail and its number means nothing. Validate a",
                  "       reference-free judge against human labels instead."]
    return "\n".join(lines)


def sweep(rows: Sequence[dict], judge: Judge, *, mode: str = "swap",
          fracs: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50),
          seed: int = 20260823) -> list[tuple[float, float]]:
    """Detection curve: how much corruption before the judge notices?"""
    return [(f, meta_evaluate(rows, judge, mode=mode, frac=f, seed=seed).delta) for f in fracs]
