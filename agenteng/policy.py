"""The permission broker: four controls, each one testable.

A guardrail you have never seen trip is a guardrail you are assuming. Every
control here exists so it can be *fired on purpose* in a test, which is the
only way to know it works. The harness already gives you the seam: every tool
call passes through a policy callable before it executes.

FOUR CONTROLS, AND WHY THESE FOUR

    allowlist   what the agent may do at all
    scope       what data it may do it to
    budget      how much it may spend before someone re-authorises
    timeout     how long any single call may take
    kill switch one flag that stops everything, checked first

They are not interchangeable, and the second one is the one people leave out.
An allowlist stops the wrong action. **A scope stops the right action pointed
at the wrong data**, which is what an injected instruction actually asks for:
an agent that legitimately needs `read_file` is not protected by being allowed
to use `read_file`. A budget stops the right action repeated forever. A
timeout stops the action that never returns. A kill switch stops all of it when
you have decided it should stop, and needs no justification beyond being set.

WHAT THIS IS NOT
It is not a sandbox. Path containment lives in `tools.py` and is enforced
after resolution so symlinks cannot smuggle a path out. This is authorisation,
which is a different question: not *can it escape*, but *should it be doing
this at all*.
"""
from dataclasses import dataclass, field

from .tools import PolicyDenied


@dataclass
class Budget:
    """A spend cap in whatever unit you are counting. Calls, tokens, dollars.

    `spent` is deliberately not private. A budget you cannot inspect mid-run
    is a budget you will only learn about from an invoice.
    """
    limit: float
    spent: float = 0.0
    unit: str = "calls"

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


@dataclass
class PermissionBroker:
    """Sits in front of every tool call and answers one question: allowed?

    Usage:

        broker = PermissionBroker(allow={"read_file"}, budget=Budget(limit=10))
        box = Toolbox(workdir=wd, policy=broker)

    The broker is callable, so it drops straight into the existing seam with
    no changes to the loop.
    """
    allow: set[str] = field(default_factory=set)
    scopes: dict[str, list[str]] = field(default_factory=dict)
    budget: Budget | None = None
    max_timeout: int | None = None
    killed: bool = False
    denials: list[tuple[str, str]] = field(default_factory=list)
    cost_of: dict[str, float] = field(default_factory=dict)

    def kill(self, reason: str = "operator stopped the agent") -> None:
        """Stop everything. Needs no justification beyond being called."""
        self.killed = True
        self._reason = reason

    def _deny(self, tool: str, why: str):
        self.denials.append((tool, why))
        raise PolicyDenied(why)

    def __call__(self, tool: str, args: dict) -> None:
        # Kill switch first, always. A kill switch checked after the budget is
        # a kill switch that can be outvoted by another control.
        if self.killed:
            self._deny(tool, getattr(self, "_reason", "killed"))

        if self.allow and tool not in self.allow:
            self._deny(tool, f"tool {tool!r} is not on the allowlist "
                             f"({', '.join(sorted(self.allow))})")

        # Scope. The control that stops an injected instruction, because the
        # tool it asks for is one the agent legitimately needs.
        for key in ("path", "file"):
            target = args.get(key)
            if target is None:
                continue
            allowed = self.scopes.get(tool)
            if allowed is not None and not any(str(target).startswith(pre) for pre in allowed):
                self._deny(tool, f"path {target!r} is outside the scope for {tool} "
                                 f"({', '.join(allowed)})")

        if self.max_timeout is not None:
            requested = args.get("timeout")
            if requested is not None and requested > self.max_timeout:
                self._deny(tool, f"timeout {requested}s exceeds the cap of {self.max_timeout}s")
            args.setdefault("timeout", self.max_timeout)

        if self.budget is not None:
            cost = self.cost_of.get(tool, 1.0)
            if self.budget.spent + cost > self.budget.limit:
                self._deny(tool, f"budget exhausted: {self.budget.spent:g}/"
                                 f"{self.budget.limit:g} {self.budget.unit} spent")
            self.budget.spent += cost

    def report(self) -> str:
        lines = ["permission broker"]
        lines.append(f"  allowlist   {', '.join(sorted(self.allow)) or '(open)'}")
        for tool, pres in sorted(self.scopes.items()):
            lines.append(f"  scope       {tool} -> {', '.join(pres)}")
        if self.budget:
            lines.append(f"  budget      {self.budget.spent:g}/{self.budget.limit:g} "
                         f"{self.budget.unit}, {self.budget.remaining:g} left")
        lines.append(f"  timeout cap {self.max_timeout if self.max_timeout is not None else '(none)'}")
        lines.append(f"  killed      {self.killed}")
        if self.denials:
            lines.append(f"  denied {len(self.denials)}:")
            for tool, why in self.denials:
                lines.append(f"    {tool}: {why}")
        else:
            lines.append("  denied 0  <- a control that never fires is a control you are assuming")
        return "\n".join(lines)
