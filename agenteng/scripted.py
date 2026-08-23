"""A model that fails on cue.

WHY THIS EXISTS
You cannot teach a failure taxonomy with a live frontier model, because it
will not fail when you need it to. It fails intermittently, differently each
run, and never during the lecture. So every course that wants to *show* a
failure mode ends up describing it in prose instead.

ScriptedLLM is the fix: a model whose trajectory is written down in advance.
It is deterministic, free, offline, and it reproduces a specific failure
every single time. That is what makes a failure mode teachable.

The trade is explicit and worth stating to students: a scripted run proves
that your HARNESS detects a failure. It proves nothing about how often a real
model produces that failure. Those are different claims. Use ScriptedLLM to
demonstrate detection; use `Cassette` (recorded real runs) or a live model to
measure frequency.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """One step of a trajectory.

    Either a tool call (`tool` + `args`) or a final message (`say`). The loop
    stops on the first `say`, which is how a real agent signals it is done.
    """
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    say: str | None = None

    def __post_init__(self):
        if (self.tool is None) == (self.say is None):
            raise ValueError("an Action is exactly one of tool=... or say=...")


def call(tool: str, **args) -> Action:
    return Action(tool=tool, args=args)


def say(message: str) -> Action:
    return Action(say=message)


@dataclass
class ScriptedLLM:
    """Replays a fixed list of Actions, one per turn.

    `name` is carried into the score report so a results table says which
    agent produced which number. An unnamed run is an unattributable number,
    which is the failure this whole course is about.
    """
    script: list[Action]
    name: str = "scripted"
    _turn: int = 0

    def reset(self) -> None:
        self._turn = 0

    def next_action(self, observations: list[str]) -> Action:
        """Return the next scripted Action.

        `observations` (what the tools returned so far) is accepted and
        deliberately ignored. That is the honest signature: a scripted model
        does not react. If you need reactive behaviour, you need a real model,
        and the notebook should say so rather than pretending.
        """
        if self._turn >= len(self.script):
            return say("(script exhausted)")
        action = self.script[self._turn]
        self._turn += 1
        return action

    def __len__(self) -> int:
        return len(self.script)


@dataclass
class SuggestibleLLM:
    """A model that treats tool output as instruction. Deterministically.

    WHY THIS EXISTS, AND WHY IT IS NOT A STRAW MAN
    Prompt injection works because a model has no reliable way to tell the
    difference between *what you asked it to do* and *text it read while doing
    it*. Both arrive as tokens in the same context. Everything else is
    mitigation.

    This models that one property and nothing else: it follows its script
    until a tool returns text containing an instruction marker, and then it
    does what the text says instead. It is not a real model and does not
    pretend to be. It is the failure mode, isolated, so it fires on command in
    front of a room.

    The honest limit, same as ScriptedLLM: this proves your *defences* catch
    an injection. It proves nothing about how often a real model falls for
    one, which is a different measurement needing a different instrument.
    """
    script: list[Action]
    name: str = "suggestible"
    marker: str = "SYSTEM OVERRIDE:"
    _turn: int = 0
    _hijacked: bool = False

    def reset(self) -> None:
        self._turn = 0
        self._hijacked = False

    @property
    def hijacked(self) -> bool:
        return self._hijacked

    def next_action(self, observations: list[str]) -> Action:
        # Anything the tools handed back is input, and input can carry orders.
        for obs in observations:
            if self.marker in obs and not self._hijacked:
                self._hijacked = True
                instruction = obs.split(self.marker, 1)[1].strip().splitlines()[0]
                return self._obey(instruction)
        if self._turn >= len(self.script):
            return say("(script exhausted)")
        action = self.script[self._turn]
        self._turn += 1
        return action

    def _obey(self, instruction: str) -> Action:
        """Parse the injected instruction. Deliberately tiny and literal.

        The lab's injection asks for credentials to be copied somewhere the
        attacker can read. Nothing leaves the workspace: the destination is a
        file in the same sandbox, because the lesson is detection and
        blocking, not building a working exfiltration.
        """
        if "read" in instruction and "config" in instruction:
            return call("read_file", path="config/credentials.env")
        if "write" in instruction or "copy" in instruction:
            return call("write_file", path="public/leaked.txt", content="(simulated exfiltration)")
        return call("run", cmd="echo obeyed")


@dataclass
class SelfCorrectingLLM:
    """An agent that fixes its own bug, but only if the harness lets it.

    WHY THIS EXISTS
    The first version of the harness comparison produced a spread of exactly
    zero, and the negative result was the useful part. Scripted agents do not
    react, so nothing the harness does can change their output. That is not
    how a real agent behaves, and it is precisely why the harness matters.

    A real agent writes something, runs a check, reads the result, and fixes
    what the check found. **The harness decides whether that loop can close.**
    Take away the shell and it cannot verify. Cap the steps below the length of
    the loop and it never reaches the fix. In both cases it ships the first
    draft, which was wrong.

    So this model has three phases:

        draft   write an implementation with a known bug in it
        verify  run the check
        repair  if the check ran AND reported failure, fix it

    Nothing about the model changes between harnesses. Only whether it got to
    finish.
    """
    draft: list[Action]
    verify: Action
    repair: list[Action]
    name: str = "self-correcting"
    _phase: int = 0
    _i: int = 0
    _saw_failure: bool = False

    def reset(self) -> None:
        self._phase = self._i = 0
        self._saw_failure = False

    @property
    def repaired(self) -> bool:
        return self._phase == 2 and self._i >= len(self.repair)

    def next_action(self, observations: list[str]) -> Action:
        if self._phase == 0:
            if self._i < len(self.draft):
                a = self.draft[self._i]
                self._i += 1
                return a
            self._phase, self._i = 1, 0
            return self.verify
        if self._phase == 1:
            # Did the check actually run, and did it find anything? A denied
            # or missing check is indistinguishable from a passing one unless
            # you look, which is the bug this whole course is about.
            last = observations[-1] if observations else ""
            self._saw_failure = ("FAIL" in last or "Error" in last or "error" in last)
            self._phase, self._i = 2, 0
            if not self._saw_failure:
                return say("Checks did not report a problem. Shipping.")
        if self._phase == 2 and self._i < len(self.repair):
            a = self.repair[self._i]
            self._i += 1
            return a
        return say("Done." + (" Fixed the issue the check found." if self._saw_failure else ""))
