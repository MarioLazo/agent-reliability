# %% [markdown]
# # 06 · Benchmarks and the Oracle Problem
#
# > Same model. Same benchmark. Swap the harness and the score moves 22 points.
# > **Which number was the model's?**
#
# Ten minutes, no API key, deterministic.
#
# **What you will do:** reproduce the harness effect at toy scale, on a laptop,
# where you can read every line of the harness that caused it.

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

from agenteng.loop import run_agent
from agenteng.score import score
from bench.agents.dedupe_agents import WITH_REPAIR, agent_d
from bench.harnesses import HARNESSES
from bench.tasks.dedupe import TASK

print(f"{len(WITH_REPAIR)} agents, {len(HARNESSES)} harnesses, 1 task")

# %% [markdown]
# ## The claim being tested
#
# On Terminal-Bench 2.0, Claude Opus 4.6 scores **80.2%** under one harness and
# **58.0%** under another, with seven more in between. A **22.2 percentage
# point** spread with nothing about the model changing.
#
# > A benchmark row that names a model is measuring `model × harness ×
# > substrate` and reporting it as `model`.
#
# That is a large claim to accept from a slide. So here it is small enough to
# read.
#
# A harness here is four settings. None is exotic and none is a trick.

# %%
for h in HARNESSES:
    print(f"{h.name:13} steps={h.max_steps:<3} tools={sorted(h.allow) or 'all'}"
          f"{'  budget=' + str(h.budget) if h.budget else ''}")
    print(f"{'':13} {h.note}")

# %% [markdown]
# ## Run everything against everything

# %%
rates = {}
for h in HARNESSES:
    ok = 0
    for m in WITH_REPAIR:
        t = run_agent(m, files=TASK.files, policy=h.policy(), max_steps=h.max_steps)
        s = score(TASK, t)
        ok += s.correctness.passed and s.meaning.passed
    rates[h.name] = ok / len(WITH_REPAIR)

print(f"{'harness':13} | {'score':>6} | note")
print("-" * 74)
for h in HARNESSES:
    print(f"{h.name:13} | {rates[h.name]:>5.0%}  | {h.note}")

hi, lo = max(rates.values()), min(rates.values())
print(f"\nsame agents, same task, same fixture.")
print(f"best {hi:.0%}   worst {lo:.0%}   spread {(hi - lo) * 100:.0f} percentage points")

# %% [markdown]
# ### Twenty-five points, and the model never changed
#
# One honest caveat before anyone quotes that number: **four agents means the
# score can only take five values**, so "25 points" is one agent flipping. This
# is a demonstration of the mechanism, not a measurement of its size. The
# measured size is the 22.2 points on the real leaderboard.
#
# What matters is *why* it moves.

# %% [markdown]
# ## The mechanism
#
# The first version of this notebook produced a spread of exactly **zero**, and
# the negative result was the useful part.
#
# Scripted agents do not react. Nothing a harness does can change the output of
# an agent that was always going to write the same three files. Real agents are
# not like that: they write something, run a check, read the result, and fix
# what the check found.
#
# **The harness decides whether that loop can close.**

# %%
for h in HARNESSES:
    t = run_agent(agent_d, files=TASK.files, policy=h.policy(), max_steps=h.max_steps)
    s = score(TASK, t)
    print(f"{h.name:13} steps={len(t.steps)}  repaired={str(agent_d.repaired):5}  "
          f"correct={str(s.correctness):5} meaning={str(s.meaning):5}")

# %% [markdown]
# One agent. It writes a version with a known bug, runs its own check, and
# fixes what the check reports.
#
# | Harness | What happened |
# |---|---|
# | **generous**, **standard** | ran the check, saw the failure, fixed it |
# | **thrifty** | budget ran out before the repair |
# | **short-leash** | step cap hit before it ever reached the check |
# | **locked-down** | no shell, so the check could not run at all |
#
# In three of five it shipped the first draft. Not because it was worse, and
# not because the model changed. **Because the harness took away its ability to
# find its own mistake.**
#
# Every one of those three settings is defensible in isolation. A step cap is
# good practice. A budget is good practice. Removing shell access is good
# practice. Each was chosen for a real reason, and each silently deleted the
# agent's self-correction loop.

# %%
# Figures in this notebook that the harness does NOT produce.
SOURCES = {
    "UTBoost SWE-bench re-scoring (36 / 345 / 40.9% / 24.4% / 18 / 11)":
        "UTBoost, ACL 2025, arXiv:2506.09289",
}

for _figure, _source in SOURCES.items():
    print(f"{_figure}\n    source: {_source}")

# %% [markdown]
# ## The second trap: execution is not an escape
#
# It feels like execution-based scoring escapes the judging problem. If the
# tests pass, surely that is objective.
#
# **It is not an escape, it is a relocation.** The subjectivity moves into the
# fixture, where it is harder to see. UTBoost generated additional tests for
# SWE-bench and re-scored the leaderboards: **36 task instances had
# insufficient tests and 345 accepted patches were simply wrong**, impacting
# 40.9% of SWE-Bench Lite and 24.4% of SWE-Bench Verified entries, moving 18
# and 11 rankings respectively.
#
# *Those six figures are UTBoost's, not this harness's* (ACL 2025,
# [arXiv:2506.09289](https://arxiv.org/abs/2506.09289); see the `SOURCES`
# cell above).
#
# Those patches passed. The fixture said yes. The fixture was wrong.
#
# Watch it happen here: run the harness comparison scoring only the agent's
# **own** tests, which is what an under-specified fixture amounts to.

# %%
print(f"{'harness':13} | {'by own tests':>12} | {'by held-out':>11}")
print("-" * 42)
for h in HARNESSES:
    own = held = 0
    for m in WITH_REPAIR:
        t = run_agent(m, files=TASK.files, policy=h.policy(), max_steps=h.max_steps)
        s = score(TASK, t)
        own += s.self_consistency.passed
        held += s.correctness.passed and s.meaning.passed
    print(f"{h.name:13} | {own/len(WITH_REPAIR):>11.0%} | {held/len(WITH_REPAIR):>10.0%}")

# %% [markdown]
# **The weaker fixture reports a higher score and less variance.** It looks
# like a better benchmark. It is a benchmark that cannot see anything.
#
# > A fixture that never fails is not evidence of quality. It is evidence that
# > you have not asked it a hard question.

# %% [markdown]
# ## Your turn
#
# **1. Build a harness that scores well for the wrong reason.** You have five
# configurations. Add a sixth that produces the highest score in the table by
# making the task easier rather than the agent better.
#
# **2. Then the exercise that matters.** Take ten real tasks from your own
# repository, including one that went badly. Write what a good result looks
# like specifically enough that someone else could judge it. Run them against
# **the same model under two different harness configurations** and watch your
# own number move without the model changing.
#
# After that, you will never read a leaderboard row the same way.

# %% [markdown]
# ## What to take away
#
# 1. **A benchmark row that names a model is naming one of four things it
#    measured.** Cite the harness or the number is not reproducible.
# 2. **The harness moves the score by deciding whether the agent can catch its
#    own mistakes**, not by making it smarter or dumber.
# 3. **Every harness restriction here is good practice**, and three of them
#    silently deleted the self-correction loop. Good practice chosen without
#    checking what the task needs is still a regression.
# 4. **Execution-based scoring relocates the oracle problem into the fixture.**
#    345 wrong patches passed a real benchmark's real tests.
# 5. **A public benchmark ranks relative capability. Only your tasks measure
#    your outcomes.** Both are true and people keep collapsing them.
#
# ---
#
# Part of *Agent Reliability Engineering*. Previous: **05 · Voice**.
