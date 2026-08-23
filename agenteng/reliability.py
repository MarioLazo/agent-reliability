"""Topology arithmetic: why adding agents makes systems worse.

    Three stages at 90% each is not a 90% system. It is 0.9 x 0.9 x 0.9,
    which is 72.9%, and that is the whole module.

The non-obvious consequence, and the reason this file exists rather than a
slide with the multiplication on it: **a verifier that catches only some
errors beats making every stage more accurate, and it costs far less.** You
can feel that claim is plausible. You cannot feel by how much, or where it
stops being true. So: run it.

Stdlib only. `random` rather than numpy, seeded, and the loop is written to be
read rather than to be fast. At the scale a lesson needs, a readable loop that
runs in under a second is worth more than a vectorised one nobody opens.
"""
import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Topology:
    """A shape, its analytic reliability, and what it costs to run.

    `cost` counts stage-executions per task, which is a proxy for tokens.
    It is deliberately crude: the point is the ratio between topologies, not
    a billing estimate.
    """
    name: str
    reliability: Callable[[float], float]
    cost: Callable[[float], float]
    note: str


def _chain(p: float, n: int = 3) -> float:
    return p ** n


def _fanout_verify(p: float, n: int = 3, recall: float = 0.8) -> float:
    """Each stage runs, then a verifier catches `recall` of the failures.

    A caught failure is retried once. The verifier is not perfect and is not
    assumed to be: `recall` is the fraction of bad outputs it notices.
    """
    per_stage = p + (1 - p) * recall * p
    return per_stage ** n


def _hierarchical(p: float, n: int = 3, retries: int = 2) -> float:
    """A coordinator re-runs a failed stage up to `retries` times.

    Note what this does NOT model: a coordinator that cannot tell a failure
    from a success. That is the realistic case and it is exactly why the
    verifier's recall matters more than the retry count.
    """
    per_stage = 1 - (1 - p) ** (retries + 1)
    return per_stage ** n


TOPOLOGIES = [
    Topology("sequential chain", _chain, lambda p: 3.0,
             "A -> B -> C. Every error propagates. The default, because it is what you get by not deciding."),
    Topology("fan-out + verifier", _fanout_verify, lambda p: 3.0 + 3.0 + 3.0 * (1 - p),
             "Each stage checked by a verifier at 80% recall, failures retried once."),
    Topology("hierarchical retry", _hierarchical, lambda p: 3.0 * (1 + (1 - p) + (1 - p) ** 2),
             "A coordinator re-runs a failed stage up to twice. Assumes it can tell."),
]


def monte_carlo(topology: str, p: float, trials: int = 20_000, seed: int = 20260823,
                n: int = 3, recall: float = 0.8, retries: int = 2) -> float:
    """Simulate the same shapes, so the analytic numbers can be checked.

    An analytic formula you have not simulated is a formula you have not
    tested, and this course is not going to teach arithmetic on trust.
    """
    rng = random.Random(seed)
    ok = 0
    for _ in range(trials):
        good = True
        for _stage in range(n):
            passed = rng.random() < p
            if not passed:
                if topology == "fan-out + verifier":
                    if rng.random() < recall and rng.random() < p:
                        passed = True
                elif topology == "hierarchical retry":
                    for _ in range(retries):
                        if rng.random() < p:
                            passed = True
                            break
            if not passed:
                good = False
                break
        ok += good
    return ok / trials


def compare(p: float = 0.9, trials: int = 20_000) -> list[dict]:
    return [{
        "topology": t.name,
        "analytic": t.reliability(p),
        "simulated": monte_carlo(t.name, p, trials=trials),
        "cost": t.cost(p),
        "note": t.note,
    } for t in TOPOLOGIES]


def table(rows: list[dict], p: float) -> str:
    w = max(len(r["topology"]) for r in rows)
    head = f"{'topology'.ljust(w)} | analytic | simulated |  cost | reliability per unit cost"
    out = [f"three stages, each {p:.0%} reliable", "", head, "-" * len(head)]
    for r in rows:
        out.append(f"{r['topology'].ljust(w)} |  {r['analytic']:.3f}   |   {r['simulated']:.3f}   | "
                   f"{r['cost']:5.2f} | {r['analytic']/r['cost']:.4f}")
    return "\n".join(out)


def better_agents_vs_verifier(p: float = 0.9, bump: float = 0.05, recall: float = 0.8) -> dict:
    """The comparison that reframes the design question.

    Left: make every stage `bump` more accurate, which is expensive, slow, and
    often not available at any price. Right: leave the stages alone and add a
    verifier at `recall`.
    """
    return {
        "baseline": _chain(p),
        "better_agents": _chain(min(p + bump, 1.0)),
        "add_verifier": _fanout_verify(p, recall=recall),
        "p": p, "bump": bump, "recall": recall,
    }
