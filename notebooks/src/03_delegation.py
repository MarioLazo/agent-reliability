# %% [markdown]
# # 03 · Delegation and Cascading Error
#
# > Two credible companies published directly opposing conclusions about
# > multi-agent systems within six months of each other. **Both are right**,
# > and the condition that separates them is the actual lesson.
#
# Ten minutes. No API key, no cost, no install, fully deterministic.
#
# **What you will do:** watch three agents each do their job correctly and
# produce a wrong system, then find out how much reliability a topology change
# buys compared to better agents.

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

from agenteng.reliability import better_agents_vs_verifier, compare, monte_carlo, table
from agenteng.tools import run_tests
from bench.agents.pipeline_agents import HELD_OUT, TICKET, run_pipeline

print("ready")

# %% [markdown]
# ## Part 1 · The arithmetic nobody does
#
# You have three stages. Each is 90% reliable. Somebody asks how reliable the
# system is.
#
# The answer people give is "about 90%". The answer is:

# %%
p = 0.9
print(f"{p} x {p} x {p} = {p**3:.3f}")

# %% [markdown]
# **72.9%.** Better than one in four tasks fails, and every individual
# component is performing exactly as specified.
#
# That is the entire reason multi-agent systems disappoint. Nothing is broken.
# The multiplication is doing it.
#
# Let's check the formula against a simulation, because an analytic result you
# have not simulated is a result you have not tested.

# %%
print(f"analytic  {p**3:.4f}")
print(f"simulated {monte_carlo('sequential chain', p, trials=50_000):.4f}")

# %% [markdown]
# ## Three topologies
#
# Same three stages, same 90%, arranged differently. `cost` counts stage
# executions per task, which is a crude proxy for tokens. The ratio between
# rows is the point, not the absolute number.

# %%
rows = compare(p, trials=50_000)
print(table(rows, p))

# %% [markdown]
# ### The result that should change what you build
#
# Look at the two ways to improve a 72.9% system.

# %%
d = better_agents_vs_verifier(p=0.9, bump=0.05, recall=0.8)
print(f"baseline, three stages at {d['p']:.0%}      {d['baseline']:.3f}")
print(f"make every stage {d['bump']:.0%} more accurate   {d['better_agents']:.3f}")
print(f"leave the stages, add a verifier   {d['add_verifier']:.3f}")

# %% [markdown]
# **A verifier that catches only 80% of errors beats making every single stage
# five percent more accurate.**
#
# And notice which one you can actually do. Making every stage 5% better means
# a better model, a better prompt, better tools, on all three, and it is often
# not available at any price. Adding a verifier is an afternoon.
#
# > **The design question is not "how do I get better agents." It is "what
# > shape should this be."** That is a question you can act on this week.

# %% [markdown]
# ### The row that is lying to you, and why we left it in
#
# `hierarchical retry` looks best on both reliability and cost. It is not.
#
# Its formula assumes the coordinator **can tell a failed stage from a
# successful one**. If it could reliably do that, you would already have solved
# the verification problem, and you would not need the retries.
#
# The topology that looks best is the one whose assumption is doing all the
# work. Read the assumptions before the numbers, every time. In this file they
# are in the docstrings, deliberately, next to the formula they support.

# %% [markdown]
# ## Part 2 · What a cascade looks like from the inside
#
# The arithmetic tells you failures multiply. It does not tell you what that
# feels like when you are reading the logs, which is the thing that actually
# stops people adding agents.
#
# Here is the ticket. Three agents will now work on it in sequence: an
# architect writes the interface, a developer implements it, a tester tests it.

# %%
print(TICKET)

# %%
workdir, trajectories = run_pipeline()
for t in trajectories:
    print(t.transcript())
    print()

# %% [markdown]
# **Read those three transcripts again and find the mistake.**
#
# The architect defined an interface. The developer implemented it faithfully,
# and added a range check the spec implied. The tester wrote four tests
# covering the happy path, both boundaries and the error case, and they pass.
#
# Every agent did its job correctly. Every handoff was clean.

# %%
print((workdir / "checkout.py").read_text())
r = run_tests(workdir, "test_checkout.py")
print("the pipeline's own tests:", "PASS" if r.ok else "FAIL")

# %% [markdown]
# Now the test written by the person who filed the ticket, which was shown to
# nobody in the chain.

# %%
(workdir / "test_reality.py").write_text(HELD_OUT)
r = run_tests(workdir, "test_reality.py")
print("what the ticket asked for:", "PASS" if r.ok else "FAIL")
print()
print("\n".join(l for l in r.output.splitlines() if "AssertionError" in l))

# %% [markdown]
# ### One word, three stages, perfect fidelity
#
# The ticket said *show what the customer pays*. The architect wrote a spec for
# a function returning the discount **amount**. `20.0`, not `80.0`.
#
# After that, everything downstream is correct **relative to its input**. The
# developer implemented the spec. The tester tested the spec. Nobody was
# careless. Nobody could have caught it by doing their own job better, because
# each of them only ever saw the output of the stage above.
#
# > **A cascade is not a chain of mistakes. It is one mistake, transmitted
# > perfectly.** And the further it travels, the more corroborating evidence it
# > accumulates: by stage three there are four passing tests agreeing with it.
#
# This is why specification and coordination account for **79% of measured
# multi-agent failures** (MAST: 41.8% specification, 36.9% inter-agent
# misalignment, from 1,600+ annotated traces). Implementation is the 21%.

# %% [markdown]
# ## Your turn
#
# **1. Put the verifier in the right place.** You now have a pipeline and a
# reliability model. Where would a verifier have caught this? Not after stage
# three: by then four tests agree with the error. Change the fixture below and
# find the only stage where a check would have worked.

# %%
# The architect's spec is the only artifact that is wrong. What would you check
# it against, given the ticket is the only other thing you have?
print("ticket :", TICKET.strip().splitlines()[-1])
print("spec   :", (workdir / "spec.py").read_text().strip().splitlines()[3])

# %% [markdown]
# **2. Sweep the verifier and find out what you are actually buying.** Move
# recall from 0.0 to 1.0 and watch two columns that do not agree.

# %%
from agenteng.reliability import _chain, _fanout_verify

print(f"{'recall':>7} | {'reliability':>11} | {'per unit cost':>13}")
print("-" * 37)
for recall in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    rel = _fanout_verify(0.9, recall=recall)
    cost = 3.0 + 3.0 + 3.0 * (1 - 0.9)
    print(f"{recall:>7.1f} | {rel:>11.3f} | {rel/cost:>13.4f}")
print(f"\nchain for comparison: {_chain(0.9):.3f} reliability at 3.00 cost "
      f"= {_chain(0.9)/3.0:.4f} per unit")

# %% [markdown]
# ### The verifier never wins on cost-efficiency. Not at any recall.
#
# Even a **perfect** verifier lands at 0.1540 per unit against the chain's
# 0.2430. The sweep has no crossover, and the first version of this exercise
# asked you to find one. There is not one.
#
# That is the finding, and it is more useful than the crossover would have
# been. **Reliability-per-unit-cost is the wrong objective whenever failure is
# expensive**, which is the entire premise of the course. The chain is the most
# efficient way to be wrong 27% of the time.
#
# So the real question is never "which topology is most efficient." It is:
#
# > **What does one failure cost, and how many can you afford?**
#
# Answer that first and the topology falls out of it. Answer it second and you
# will optimise your way to a cheap system nobody trusts.

# %% [markdown]
# ## What to take away
#
# 1. **Three stages at 90% is a 73% system.** The multiplication, not a bug.
# 2. **A partial verifier beats better agents**, and it is the one you can
#    actually build this week.
# 3. **Read the assumption before the number.** The best-looking topology here
#    assumes away the hardest problem in the course.
# 4. **A cascade is one mistake transmitted perfectly**, and it gains
#    corroborating evidence as it travels. Four passing tests agreed with it.
# 5. **Start with two agents and earn the third.** Anthropic measured
#    multi-agent at roughly **15x the tokens** of chat, paying off on
#    breadth-first exploration and losing where agents share context, with
#    "most coding tasks" named as unsuitable. Cognition's *Don't Build
#    Multi-Agents* argues for single-threaded by default. Both are right, and
#    15x is a buyer's argument that is much harder to wave away than an
#    engineering one.
#
# ---
#
# Part of *Agent Reliability Engineering*. Previous: **02 · Judging the Judge**.
