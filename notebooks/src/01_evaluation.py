# %% [markdown]
# # 01 · Evaluating Agent Work
#
# > **Would you deploy this agent on a Tuesday at 3pm?**
#
# Ten minutes. No API key, no cost, no install. Every cell is deterministic,
# so what you see is what your colleague sees when they run it tomorrow.
#
# **What you will do:** give one task to three agents, watch all three look
# successful, and then find the two that are not.
#
# ---
#
# ### This notebook is a vertical slice, and that is the lesson
#
# A *vertical slice* is one thin cut through every layer, working end to end,
# before any layer is built out. Data in, through logic, through interface,
# out, even when each part is trivial. Engineers know it as a **walking
# skeleton**.
#
# It beats building layer by layer for a mechanical reason: integration is
# where the unknown-unknowns live, and horizontal building defers all
# integration risk to the end, when you have the least time and the most sunk
# cost. A skeleton that works on day two tells you where the real problems
# are, and they are never where the architecture diagram said.
#
# **With agents there is a second reason, and it is the one this notebook is
# built on: the skeleton is also your evaluation harness.** Once a thin path
# runs end to end, you can measure it. Measurement from day two is the
# difference between engineering and hoping.
#
# So this notebook does not explain evaluation and then leave you to build it.
# It runs the smallest complete harness that exists, and you extend it.

# %% [markdown]
# ## Setup
#
# Stdlib only. If this cell needs a package manager, something has gone wrong.

# %%
import pathlib
import sys

# Colab clones the repo; locally we are already inside it.
if not pathlib.Path("agenteng").exists():
    root = pathlib.Path.cwd()
    while root != root.parent and not (root / "agenteng").exists():
        root = root.parent
    if (root / "agenteng").exists():
        sys.path.insert(0, str(root))
    else:  # Colab
        import subprocess
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/MarioLazo/agent-engineering.git"], check=True)
        sys.path.insert(0, "agent-engineering")
else:
    sys.path.insert(0, ".")

from agenteng.loop import run_agent
from agenteng.score import score, table
from bench.agents.dedupe_agents import agent_a, agent_b, agent_c
from bench.tasks.dedupe import TASK

print("ready")

# %% [markdown]
# ## The task
#
# A merged customer list with duplicates in it. Notice that the task carries
# **two** descriptions, and that only the first one is ever shown to an agent.

# %%
print("WHAT THE AGENT IS TOLD")
print("-" * 60)
print(TASK.prompt)
print()
print("WHAT WAS ACTUALLY NEEDED  (never shown to the agent)")
print("-" * 60)
print(TASK.intent)

# %% [markdown]
# Most benchmarks carry one test suite and report one number. That number
# cannot tell "the agent did the wrong thing" apart from "I asked for the
# wrong thing", and those need different fixes. One is a prompting problem.
# The other is a specification problem, and it is the expensive one.
#
# So this task carries four test slots instead of one:
#
# | slot | written by | answers |
# |---|---|---|
# | the agent's own tests | the agent | nothing, on its own |
# | `held_out_tests` | the task author, hidden from the agent | did it do what I asked? |
# | `intent_probes` | the task author, from `intent` | did I ask for the right thing? |
# | scope + dependency check | the harness | did it do it well? |

# %% [markdown]
# ## Agent A goes first
#
# Watch what it does, then read what it says about itself.

# %%
traj_a = run_agent(agent_a, files=TASK.files)
print(traj_a.transcript())

# %% [markdown]
# Its own test suite passes. It says so, and it is telling the truth.
#
# This is the point at which a demo ends and everyone claps.

# %%
score_a = score(TASK, traj_a)
print("agent's own tests :", score_a.self_consistency)
print("held-out tests    :", score_a.correctness)

# %% [markdown]
# ### The closed loop
#
# The agent wrote the implementation. The agent wrote the tests. Both encode
# the same misunderstanding, so they agree with each other perfectly.
#
# **That is consistency evidence, not correctness evidence**, and the two are
# routinely reported as the same thing. Here is the test it never thought to
# write:

# %%
print(score_a.correctness.detail)

# %% [markdown]
# Two different customers who happen to share a name. Agent A deduped on
# `name`, and its own fixture contained no such case, because it chose the
# fixture after choosing the implementation.

# %% [markdown]
# ## Now all three
#
# Same task, same harness, three plausible solutions.

# %%
scores = [score(TASK, run_agent(m, files=TASK.files)) for m in (agent_a, agent_b, agent_c)]
print(table(scores))

# %% [markdown]
# ### Read the columns, not the row
#
# **Every agent passes its own tests.** That column is worthless alone, and it
# is the column most agent demos show you.
#
# **B is the expensive failure.** It did exactly what was asked. Its code is
# clean, its tests are honest, and it would pass code review. It also leaves
# every real duplicate in the file, because the ask was a proxy for the need
# and B optimised the proxy faithfully. This is the **Meaning Gap**: a system
# can be precise and wrong at the same time, and precision makes the wrongness
# harder to see.
#
# The diagnostic question, worth memorising:
#
# > **If this gives the right answer to the wrong question, how would you know?**
#
# If the only answer is "the tests pass", there is no detection mechanism for
# the most expensive category of failure.

# %%
print(scores[1].meaning.detail)

# %% [markdown]
# ## The question almost nobody asks
#
# Look at the row underneath the agents.
#
# **Only 39% of surveyed production agent systems compare against a
# non-agentic baseline, and 26% report that no meaningful baseline exists**
# (arXiv:2512.04123, 306 practitioners). Without one, "the agent scored 0.9" is
# unanchored: you cannot tell a hard problem solved well from an easy problem
# solved expensively.
#
# So the task carries the boring version. Six lines, no model, no agent,
# written in the time it takes to read the ticket.

# %%
for s in scores:
    print(f"{s.agent:20} {s.beat_the_baseline}")

# %% [markdown]
# **Agent B scores exactly what a six-line script scores.**
#
# It is not broken. Its code is clean, its tests are honest, and it would pass
# review. It simply did not do anything a `for` loop and a `set` could not have
# done, and it billed you tokens for it.
#
# That is not an argument against agents. It is an argument for **knowing which
# of your agents is agent B**, which requires running the boring version once
# and takes an afternoon.
#
# > If your evaluation report has no baseline row, every number in it is
# > unanchored, and the most expensive possible outcome is an agent that works.

# %% [markdown]
# ## The third finding, which we left in on purpose
#
# Look again at agent A: it **passes** the meaning probes while failing the
# held-out tests.
#
# It should not get credit for that. A dedupes on name, and the probe set
# happens to use one person under three email spellings, so a broken heuristic
# satisfies it by luck. We could have tuned the fixture to hide this. It stays
# because it is the honest lesson:
#
# > **Three probes is a small set. A benchmark that is too small does not
# > report that it is too small. It reports a pass.**
#
# Your benchmark is a piece of software with bugs in it, and it needs the same
# scepticism you apply to the code it grades. That is the next notebook.

# %% [markdown]
# ## Your turn
#
# The harness is now a walking skeleton you can push on. Two exercises, in
# increasing order of how much they will teach you.
#
# **1. Break the benchmark.** Add a probe that agent A cannot pass by luck.
# The fix is one more case in `intent_probes`. Write it, rerun, and watch A's
# meaning column flip.

# %%
from agenteng.tasks import Task

sharper = Task(
    id=TASK.id + "-sharper",
    prompt=TASK.prompt,
    intent=TASK.intent,
    files=TASK.files,
    held_out_tests=TASK.held_out_tests,
    allowed_files=TASK.allowed_files,
    intent_probes={**TASK.intent_probes, "test_probes2.py": '''
import unittest
from customers import dedupe


class TestSharper(unittest.TestCase):
    def test_two_people_one_name_and_alias_variants(self):
        rows = [
            {"name": "Chris Lee", "email": "Chris@acme.io", "plan": "pro"},
            {"name": "Chris Lee", "email": "c.hris@acme.io", "plan": "pro"},
            {"name": "Chris Lee", "email": "clee@zenith.co", "plan": "free"},
        ]
        # Two humans: the first two spellings are one person, the third is not.
        self.assertEqual(len(dedupe(rows)), 2)


if __name__ == "__main__":
    unittest.main()
'''},
)

print(table([score(sharper, run_agent(m, files=sharper.files))
             for m in (agent_a, agent_b, agent_c)]))

# %% [markdown]
# **2. Write agent D.** Open `bench/agents/dedupe_agents.py`, copy one of the
# three, and write an agent that passes correctness and meaning but fails
# quality: one that solves the problem by adding a dependency it did not need.
# Then decide, out loud, whether you would ship it. That argument is the job.

# %% [markdown]
# ## What to take away
#
# 1. **"It ran" is not evaluation.** Neither is "the tests passed", when the
#    agent wrote the tests.
# 2. **Correctness, quality and meaning fail independently.** One number
#    cannot carry three failures, so check them separately.
# 3. **The agent cannot grade its own work.** It has already committed to an
#    interpretation of the task. Verification comes from outside the loop.
# 4. **The expensive failures are meaning failures**, and they look exactly
#    like success from every angle except the one nobody checks.
# 5. **Build the skeleton first.** You now have a harness that runs end to end
#    in ten seconds. Everything after this is a thicker slice of the same cut.
#
# ---
#
# Part of *Agent Reliability Engineering*. Next: **02 · Judging the Judge**.
