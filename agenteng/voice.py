"""Voice reliability, simulated. Four models, zero dependencies.

WHAT THIS MODELS, AND WHAT IT DOES NOT
It models the **control logic** of a voice agent: where the latency goes, when
a barge-in fires, how much an utterance actually tells you, and what a
confirmation protocol leaves behind. All of that is arithmetic and policy, and
all of it can be wrong in ways you can reason about offline.

It does **not** model acoustics. There is no audio here, no ASR, no TTS, no
speaker. A simulated barge-in fires because a number crossed a threshold, not
because someone spoke. That is the same honest limit as `ScriptedLLM`: this
proves your *logic* behaves as designed, never that a real caller in a real
truck cab gets the outcome you wanted.

Going live is described at the bottom of this file, along with what it costs.

Stdlib only, seeded, deterministic.
"""
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Latency budget
# ---------------------------------------------------------------------------
# Human conversational turns land in roughly 200-400ms. Under ~500ms an agent
# reads as composed; past ~600ms it reads as slow; past ~1200ms at p50 callers
# start asking "hello? are you there?" and talking over it.
#
# The number that matters is p95, not p50. A caller does not average their
# experience. They remember the pause.

HUMAN_TURN_MS = (200, 400)
FEELS = [(500, "composed"), (600, "acceptable"), (1200, "slow"),
         (float("inf"), "broken: callers will talk over it")]


@dataclass
class Stage:
    """One component of the round trip, with its own spread."""
    name: str
    p50: int
    p95: int
    note: str = ""


# A deliberately ordinary pipeline. The point is what it adds up to.
DEFAULT_PIPELINE = [
    Stage("endpointing", 250, 400, "waiting long enough to be sure they stopped"),
    Stage("ASR final", 150, 350, "partial results are earlier; the final is what you act on"),
    Stage("intent + state lookup", 80, 200, "the part everyone forgets to budget"),
    Stage("model first token", 350, 900, "the only stage anyone talks about"),
    Stage("TTS first byte", 120, 300, ""),
    Stage("network", 60, 180, "twice, if your ASR and TTS are different vendors"),
]


@dataclass
class LatencyBudget:
    stages: list[Stage] = field(default_factory=lambda: list(DEFAULT_PIPELINE))

    def total(self, pct: str = "p50") -> int:
        return sum(getattr(s, pct) for s in self.stages)

    def feel(self, pct: str = "p50") -> str:
        t = self.total(pct)
        return next(label for cap, label in FEELS if t <= cap)

    def share(self, pct: str = "p50") -> list[tuple[str, int, float]]:
        t = self.total(pct)
        return [(s.name, getattr(s, pct), getattr(s, pct) / t) for s in self.stages]

    def report(self) -> str:
        w = max(len(s.name) for s in self.stages)
        out = [f"{'stage'.ljust(w)} |   p50 |   p95 | share of p50"]
        out.append("-" * len(out[0]))
        for name, ms, frac in self.share("p50"):
            p95 = next(s.p95 for s in self.stages if s.name == name)
            bar = "#" * round(frac * 30)
            out.append(f"{name.ljust(w)} | {ms:5d} | {p95:5d} | {bar} {frac:.0%}")
        out.append("-" * len(out[0]))
        out.append(f"{'TOTAL'.ljust(w)} | {self.total('p50'):5d} | {self.total('p95'):5d} |")
        out.append("")
        out.append(f"p50 {self.total('p50')}ms: {self.feel('p50')}")
        out.append(f"p95 {self.total('p95')}ms: {self.feel('p95')}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. Barge-in / turn-taking
# ---------------------------------------------------------------------------
# Three signals gate an interruption: is the sound loud enough, does it
# classify as voice, and has it lasted long enough to be a sentence rather
# than a cough. Tighten them and you stop cutting the agent off for no
# reason; tighten them too far and a caller who genuinely interrupts is
# ignored, which is worse.

@dataclass
class BargeInPolicy:
    energy_dbfs: float = -40.0   # typical tuning sits between -45 and -35
    require_voice_class: bool = True
    min_duration_ms: int = 250   # 200-300 is the usual guard


@dataclass
class SoundEvent:
    """One thing the microphone heard while the agent was speaking."""
    kind: str          # "speech" (a real interruption) or "noise"
    energy_dbfs: float
    duration_ms: int
    classified_voice: bool


def cab_noise(n: int = 400, seed: int = 20260823) -> list[SoundEvent]:
    """A truck cab: engine drone, road, radio, and the driver actually talking.

    Deliberately hostile. A quiet office would make every policy look good,
    which is how voice agents get tuned in the wrong room.
    """
    rng = random.Random(seed)
    events = []
    for _ in range(n):
        if rng.random() < 0.25:
            events.append(SoundEvent("speech", rng.uniform(-38, -20),
                                     rng.randint(300, 1500), True))
        else:
            kind = rng.choice(["engine", "road", "radio", "cough", "door"])
            # Radio is the nasty one: it is loud and it classifies as voice.
            voice_like = kind == "radio"
            energy = rng.uniform(-50, -22) if kind in ("radio", "door") else rng.uniform(-60, -35)
            events.append(SoundEvent("noise", energy, rng.randint(80, 900), voice_like))
    return events


def evaluate_bargein(events: list[SoundEvent], policy: BargeInPolicy) -> dict:
    fired_false = fired_true = missed = 0
    for e in events:
        fires = (e.energy_dbfs >= policy.energy_dbfs
                 and e.duration_ms >= policy.min_duration_ms
                 and (e.classified_voice or not policy.require_voice_class))
        if fires and e.kind == "noise":
            fired_false += 1
        elif fires and e.kind == "speech":
            fired_true += 1
        elif not fires and e.kind == "speech":
            missed += 1
    speech = fired_true + missed
    noise = len(events) - speech
    return {
        "false_barge_in_rate": fired_false / noise if noise else 0.0,
        "missed_interruption_rate": missed / speech if speech else 0.0,
        "false_count": fired_false, "missed_count": missed,
        "speech_events": speech, "noise_events": noise,
    }


# ---------------------------------------------------------------------------
# 3. Intent under uncertainty
# ---------------------------------------------------------------------------
# The claim this exists to prove: **the intent is not in the utterance.**
# A driver says "I'm not going to make it." Four different things that could
# mean, and the words do not separate them. State does.

INTENTS = ["late_for_delivery_window", "out_of_hours_service",
           "mechanical_breakdown", "cannot_reach_consignee"]


@dataclass
class CallState:
    """What the system already knows when the phone rings."""
    minutes_to_window_close: int
    hours_of_service_remaining: float
    miles_to_stop: float
    engine_fault_code: bool
    consignee_calls_failed: int


def from_utterance_only(utterance: str) -> dict[str, float]:
    """A weak text prior. Deliberately weak, because that is the honest case.

    Real systems use a classifier here. It does not matter: no classifier
    recovers information the sentence does not contain.
    """
    u = utterance.lower()
    scores = {i: 1.0 for i in INTENTS}
    for kw, intent in [("hours", "out_of_hours_service"), ("log", "out_of_hours_service"),
                       ("engine", "mechanical_breakdown"), ("truck", "mechanical_breakdown"),
                       ("answer", "cannot_reach_consignee"), ("dock", "cannot_reach_consignee"),
                       ("late", "late_for_delivery_window"), ("window", "late_for_delivery_window")]:
        if kw in u:
            scores[intent] += 3.0
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def with_state(utterance: str, state: CallState) -> dict[str, float]:
    """Multiply the text prior by what the system already knows.

    Plain Bayes with hand-set likelihoods. The technique is not the lesson;
    the lesson is that the state carries most of the signal and it was sitting
    in your database the whole time.
    """
    prior = from_utterance_only(utterance)
    lik = {
        "late_for_delivery_window": 3.0 if state.minutes_to_window_close < 60 else 0.4,
        "out_of_hours_service": 3.0 if state.hours_of_service_remaining < 1.0 else 0.3,
        "mechanical_breakdown": 4.0 if state.engine_fault_code else 0.2,
        "cannot_reach_consignee": 3.0 if state.consignee_calls_failed >= 2 else 0.3,
    }
    post = {k: prior[k] * lik[k] for k in INTENTS}
    total = sum(post.values())
    return {k: v / total for k, v in post.items()}


def top(dist: dict[str, float]) -> tuple[str, float]:
    k = max(dist, key=dist.get)
    return k, dist[k]


def margin(dist: dict[str, float]) -> float:
    """Gap between best and second. Below ~0.15 you should be asking, not acting."""
    ordered = sorted(dist.values(), reverse=True)
    return ordered[0] - ordered[1]


# ---------------------------------------------------------------------------
# 4. Readback
# ---------------------------------------------------------------------------
# Aviation mandated read-back and hear-back decades ago, for exactly this
# problem: voice, noise, accents, workload, and consequences. It works, and it
# leaks. Controllers catch about 90% of pilot readback errors en route, 63% in
# a tower, and 50% on radar approach. One to two percent of utterances carry
# an error to begin with.
#
# That last set of numbers is the ceiling on what confirmation buys you, and
# it is worth looking at before deciding a read-back is sufficient.

DETECTION = {"en_route": 0.90, "tower": 0.63, "radar_approach": 0.50}


def readback_residual(exchanges: int, error_rate: float = 0.015,
                      detection: float = 0.90, seed: int = 20260823) -> dict:
    rng = random.Random(seed)
    introduced = caught = escaped = 0
    for _ in range(exchanges):
        if rng.random() < error_rate:
            introduced += 1
            if rng.random() < detection:
                caught += 1
            else:
                escaped += 1
    return {"exchanges": exchanges, "introduced": introduced,
            "caught": caught, "escaped": escaped,
            "escape_rate": escaped / exchanges if exchanges else 0.0}


# ---------------------------------------------------------------------------
# Going live, and what it costs
# ---------------------------------------------------------------------------
LIVE_PATH = """
Everything above is simulated and free. A live voice agent is not, and the
honest version of this module says so rather than implying the laptop version
is the whole job.

Going live replaces four things and adds a fifth:

  endpointing   a real VAD over a real audio stream, not a duration field
  ASR           a streaming speech-to-text service.        REQUIRES AN API KEY
  model         a streaming completion.                     REQUIRES AN API KEY
  TTS           a streaming speech synthesiser.             REQUIRES AN API KEY
  telephony     a carrier or SIP trunk to receive the call. REQUIRES AN ACCOUNT

What carries over unchanged: the latency budget, the barge-in policy, the
intent-plus-state model and the readback protocol. Those are logic, and they
are the part you can get right before spending anything.

What does not carry over: any claim about accuracy. Word error rate under
engine noise, accent handling, and how a real caller reacts to a 900ms pause
are measurements, and they need real audio, real callers and a real bill.

Budget for that separately and do not let a green simulation stand in for it.
"""
