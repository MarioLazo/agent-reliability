# %% [markdown]
# # 05 · Voice: Latency, Barge-In, Intent and Readback
#
# > A driver calls dispatch from the cab. Hands on the wheel, engine at 70dB,
# > one shot at the sentence. **No backspace, no scroll-back, no reading it
# > again.**
#
# Ten minutes. No API key, no audio, fully deterministic.
#
# **What this models:** the control logic of a voice agent. Where the latency
# goes, when a barge-in fires, how much an utterance actually tells you, and
# what a confirmation protocol leaves behind.
#
# **What it does not model:** acoustics. There is no audio here. A barge-in
# fires because a number crossed a threshold, not because someone spoke. This
# proves your logic behaves as designed, never that a real caller in a real cab
# gets the outcome you wanted. The last section says what going live costs.

# %%
import pathlib
import sys

if not pathlib.Path("agenteng").exists():
    root = pathlib.Path.cwd()
    while root != root.parent and not (root / "agenteng").exists():
        root = root.parent
    sys.path.insert(0, str(root))
else:
    sys.path.insert(0, ".")

from agenteng.voice import (DETECTION, LIVE_PATH, BargeInPolicy, CallState, LatencyBudget,
                            Stage, cab_noise, evaluate_bargein, from_utterance_only, margin,
                            readback_residual, top, with_state)

print("ready")

# %% [markdown]
# ## 1 · Latency is a correctness property
#
# In text, a slow agent is annoying. In voice, a slow agent gets talked over,
# and now you have two people speaking and no transcript worth having.
#
# Human conversational turns land in **200 to 400ms**. Under ~500ms an agent
# reads as composed. Past ~600ms it reads as slow. Past ~1,200ms at p50,
# callers start saying *"hello? are you there?"*
#
# Here is an entirely ordinary pipeline. Nothing exotic, nothing badly built.

# %%
budget = LatencyBudget()
print(budget.report())

# %% [markdown]
# ### The model is 35% of the problem
#
# Everyone optimises the model. It is a third of the budget.
#
# **Endpointing alone is a quarter**, and that is just waiting long enough to be
# confident the human stopped talking. Set it too short and you interrupt them
# mid-sentence. Set it too long and you are slow. There is no setting that is
# fast and safe, only a trade you have chosen or inherited.
#
# Try the fantasy version: an infinitely fast model, zero tokens, no thinking.

# %%
free_model = LatencyBudget([s for s in budget.stages if s.name != "model first token"])
print(f"with a model that responds instantly: {free_model.total('p50')}ms p50 "
      f"-> {free_model.feel('p50')}")

# %% [markdown]
# **Still slow.** You could make the model free and instant and the agent would
# still feel sluggish, because the latency was never mostly the model.
#
# > If your voice agent is slow and your plan is a faster model, you have
# > budgeted a third of the problem.

# %% [markdown]
# ## 2 · Barge-in in a room that is not an office
#
# Three signals gate an interruption: **energy** (is it loud enough),
# **voice classification** (does it sound like speech), and a **minimum
# duration guard** (has it lasted long enough to be a sentence rather than a
# cough).
#
# The industry target is a false barge-in rate under 2%. Here is a truck cab:
# engine drone, road noise, the radio, doors, coughs, and the driver actually
# talking a quarter of the time.

# %%
events = cab_noise()
print(f"{len(events)} sound events while the agent was speaking\n")
print(f"{'policy':14} | {'false barge-in':>14} | {'missed real interruptions':>25}")
print("-" * 60)
for label, pol in [("permissive", BargeInPolicy(-50, False, 100)),
                   ("typical", BargeInPolicy(-40, True, 250)),
                   ("conservative", BargeInPolicy(-30, True, 500))]:
    r = evaluate_bargein(events, pol)
    print(f"{label:14} | {r['false_barge_in_rate']:>13.1%} | {r['missed_interruption_rate']:>24.1%}")

# %% [markdown]
# ### None of these is acceptable, and that is the finding
#
# The typical tuning sits at **10% false barge-in**, five times the target. The
# conservative tuning gets it to 3.5% and then **misses nearly half of the real
# interruptions**, which is far worse: a driver saying *"no, wrong load"* and
# being ignored is a worse failure than the agent being cut off by a door slam.
#
# **The environment beat the policy.** Three knobs are not enough in a cab, and
# no amount of tuning makes them enough. You need a fourth signal, and the
# obvious one is that the agent knows what it is currently saying, so it can
# discount its own audio.
#
# The lesson generalises past voice: **when a tuning sweep has no acceptable
# point, stop sweeping and add a signal.**

# %% [markdown]
# ## 3 · The intent is not in the utterance
#
# A driver says four words. Dispatch has to route the call.

# %%
utterance = "I'm not going to make it"
text_only = from_utterance_only(utterance)

print(f'"{utterance}"\n')
for k, v in sorted(text_only.items(), key=lambda x: -x[1]):
    print(f"  {v:5.2f}  {k}")
print(f"\ntop guess margin over second place: {margin(text_only):.2f}")

# %% [markdown]
# **A four-way tie.** The words do not separate the intents, and no classifier
# recovers information the sentence does not contain. Buying a better NLU model
# does not fix this, because it is not a model problem.
#
# It is a **specification** problem: you asked the wrong thing to carry the
# answer. Now add what the system already knew when the phone rang.

# %%
for name, state in [
    ("engine fault code active", CallState(180, 6.0, 40, True, 0)),
    ("42 minutes of drive time left", CallState(200, 0.4, 30, False, 0)),
    ("3 failed calls to the consignee", CallState(240, 5.0, 5, False, 3)),
]:
    d = with_state(utterance, state)
    intent, p = top(d)
    print(f"{name:32} -> {intent:26} p={p:.2f}  margin={margin(d):.2f}")

# %% [markdown]
# Same four words. Three different intents, each one confident.
#
# > **The state was in your database the whole time.** The utterance was never
# > going to carry it, and the effort spent on a better intent classifier is
# > effort spent on the wrong half of the problem.
#
# The `margin` is the part to operationalise. Below about 0.15 the agent should
# be **asking**, not acting, and that threshold is yours to defend with a curve
# exactly like the approval-gate one.

# %% [markdown]
# ## 4 · Readback, and its ceiling
#
# Aviation solved voice confirmation under noise, accent variation and
# workload decades ago. Read-back and hear-back is mandatory, trained, and
# audited.
#
# **It still leaks.** Controllers catch about **90%** of pilot readback errors
# en route, **63%** in a tower, and **50%** on radar approach. One to two
# percent of utterances carry an error to begin with.

# %%
print(f"{'environment':16} | {'detection':>9} | {'escaped':>7} | {'escape rate':>11}")
print("-" * 52)
for env, det in DETECTION.items():
    r = readback_residual(500, detection=det)
    print(f"{env:16} | {det:>8.0%} | {r['escaped']:>7} | {r['escape_rate']:>10.2%}")

# %% [markdown]
# Over one shift the numbers look survivable. Run a year.

# %%
YEAR = 500 * 250
for env, det in DETECTION.items():
    r = readback_residual(YEAR, detection=det)
    print(f"{env:16} {r['introduced']:>5} errors introduced, "
          f"{r['escaped']:>4} reached the driver")

# %% [markdown]
# ### What this is for
#
# Not to argue against readback. Readback is the single best control available
# and it catches most of what it sees.
#
# It is to stop **readback being treated as closure**. Confirming an
# instruction does not make it correct; it makes it *less often wrong*, at a
# rate you can now put a number on. Trained professionals in a safety-critical
# domain with a mandated protocol still let errors through, and half of them in
# the hardest environment.
#
# > **If a wrong outcome is unacceptable, a readback is not your last line of
# > defence. It is your second-to-last.** Decide what the last one is.

# %% [markdown]
# ## Your turn
#
# **1. Find a latency budget that is not slow.** Adjust the stages until p50
# lands under 500ms. You will find you cannot get there by touching the model.

# %%
tuned = LatencyBudget([
    Stage("endpointing", 150, 250, "shorter guard, more mid-sentence cuts"),
    Stage("ASR final", 100, 220, "streaming partials, act on the final"),
    Stage("intent + state lookup", 40, 90, "cache the state before the call connects"),
    Stage("model first token", 180, 400, "smaller model for routing, big one only when needed"),
    Stage("TTS first byte", 90, 200, ""),
    Stage("network", 40, 110, "same vendor for ASR and TTS"),
])
print(f"tuned: {tuned.total('p50')}ms p50 ({tuned.feel('p50')}), "
      f"{tuned.total('p95')}ms p95 ({tuned.feel('p95')})")

# %% [markdown]
# Note what each of those costs. A shorter endpointing guard cuts people off
# mid-sentence. A smaller routing model is a worse router. **Every millisecond
# here was bought from somewhere**, which is the honest version of a latency
# budget and the reason it is called a budget.
#
# **2. Add the fourth barge-in signal.** The agent knows what it is saying.
# Use that to discount its own audio and see whether you can get under 2%
# without missing real interruptions.

# %% [markdown]
# ## Going live, and what it costs

# %%
print(LIVE_PATH)

# %% [markdown]
# ## What to take away
#
# 1. **Latency is correctness in voice**, and the model is about a third of it.
# 2. **When a tuning sweep has no acceptable point, add a signal.** Three knobs
#    do not tame a truck cab.
# 3. **The intent is not in the utterance.** It is in the utterance plus the
#    state you already had.
# 4. **Readback has a measurable ceiling.** It is a very good control and it is
#    not closure.
# 5. **Simulation proves your logic, not your acoustics.** Budget separately
#    for the measurements that need real audio, real callers and a real bill.
#
# ---
#
# Part of *Agent Reliability Engineering*, Part 3. Previous: **04 · Guardrails**.
