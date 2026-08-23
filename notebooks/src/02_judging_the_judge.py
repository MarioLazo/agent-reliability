# %% [markdown]
# # 02 · Judging the Judge
#
# > You replaced human review with an LLM judge. **Who reviews the judge?**
#
# Ten minutes. No API key, no cost, no install, fully deterministic.
#
# **What you will do:** run the standard meta-evaluation test on three judges,
# discover that the standard test can pass a judge that is not reading
# anything, and find the version that cannot be fooled.
#
# ---
#
# ### The test everyone quotes
#
# > *Flip the labels on 10% of your golden set. If your judge's aggregate score
# > does not move, it is not reading the answer.*
#
# It is the software-testing move applied one level up. Mutation testing breaks
# the code and asks whether the tests notice. This breaks the reference and asks
# whether the judge notices. Meta industrialised the first at 9,095 mutants.
# Almost nobody ships the second.
#
# We are going to run it, and then we are going to break it.

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

from agenteng.metaeval import (blind_judge, honest_judge, load, meta_evaluate,
                               overlap_judge, sweep, table)

rows = load("bench/goldsets/support-answers.jsonl")
n_pass = sum(r["label"] == "pass" for r in rows)
print(f"{len(rows)} rows: {n_pass} pass, {len(rows) - n_pass} fail")

# %% [markdown]
# ## Three judges
#
# | Judge | What it does |
# |---|---|
# | **honest** | reads the reference, compares to it |
# | **overlap** | reads the reference, but only for token overlap. The realistic middle |
# | **blind** | never looks at the reference. Passes anything that looks like an answer |
#
# `blind` is not a straw man. A judge prompted *"is this a good response?"* with
# no reference in its context behaves exactly like this, and that prompt is
# extremely common.

# %%
JUDGES = [("honest", honest_judge), ("overlap", overlap_judge), ("blind", blind_judge)]

from agenteng.metaeval import agreement
for name, j in JUDGES:
    print(f"{name:8} agreement on the clean set: {agreement(j(rows), rows):.3f}")

# %% [markdown]
# **Stop here and look at `blind`.**
#
# It scores 0.700 without reading a single reference. On a golden set that is
# mostly passes, which describes nearly every golden set anyone curates, a judge
# that says yes to everything looks like a judge that is 70% right.
#
# If your evaluation report contains one number, this is the number, and you
# would ship it.

# %% [markdown]
# ## The headline test
#
# Corrupt 20% of the labels and see which judges notice.

# %%
print(table([meta_evaluate(rows, j, mode="flip", frac=0.20, name=n) for n, j in JUDGES]))

# %% [markdown]
# That looks decisive. `honest` loses exactly what it should, `overlap` loses
# half of it, and `blind` does not move at all.
#
# **Now run the identical test with a different random seed.**

# %%
print(f"{'seed':>10} | {'honest':>7} | {'overlap':>7} | {'blind':>7}")
print("-" * 42)
for s in (1, 7, 42, 2026, 20260823, 99999):
    d = [meta_evaluate(rows, j, mode="flip", frac=0.20, seed=s).delta for _, j in JUDGES]
    print(f"{s:>10} | {d[0]:+.3f} | {d[1]:+.3f} | {d[2]:+.3f}")

# %% [markdown]
# ### The headline test has a hole in it
#
# **On seed 42, the blind judge moves `+0.200`, exactly as much as the honest
# judge.** Run the standard test on that seed and you conclude your judge is
# reading the reference. It is not. It has never read a reference in its life.
#
# The mechanism is simple once you see it. Flipping a label changes the gold
# answer, not the input. A judge with fixed verdicts loses agreement on every
# flipped row it used to agree with, and gains it back on every flipped row it
# used to get wrong. **Whether that nets out to zero depends entirely on which
# rows the sampler happened to pick.**
#
# So `flip` measures *"does this judge agree with the labels"*, which is not the
# question. It also means `flip`'s discriminating power scales with the judge's
# baseline accuracy: against a coin-flip judge, expected movement is zero and
# the test cannot tell a bad judge from a lucky one.

# %% [markdown]
# ## The version that cannot be fooled
#
# `swap` keeps every label and permutes **which reference goes with which
# output**. A judge that reads *this* reference now produces different verdicts,
# so its score falls. A judge that ignores the reference produces identical
# verdicts against identical labels, so its score moves by exactly zero.

# %%
print(table([meta_evaluate(rows, j, mode="swap", frac=0.20, name=n) for n, j in JUDGES]))
print()
print(f"{'seed':>10} | {'honest':>7} | {'overlap':>7} | {'blind':>7}")
print("-" * 42)
for s in (1, 7, 42, 2026, 20260823, 99999):
    d = [meta_evaluate(rows, j, mode="swap", frac=0.20, seed=s).delta for _, j in JUDGES]
    print(f"{s:>10} | {d[0]:+.3f} | {d[1]:+.3f} | {d[2]:+.3f}")

# %% [markdown]
# **Zero on every seed, for the blind judge only.** Not approximately zero.
# Exactly zero, because permuting something the judge never reads cannot change
# what it says.
#
# That is the property you want from a test: **a result that cannot happen by
# luck.**
#
# Run both. `flip` is the one people know and it is worth reporting. `swap` is
# the one that answers the question in the headline.

# %% [markdown]
# ## How much corruption before it notices?
#
# A single number tells you pass or fail. The curve tells you the margin.

# %%
for name, j in JUDGES:
    pts = sweep(rows, j, mode="swap")
    line = "  ".join(f"{int(f*100):>2}%:{d:+.3f}" for f, d in pts)
    print(f"{name:8} {line}")

# %% [markdown]
# `honest` degrades smoothly as more references are scrambled. `blind` is a flat
# line at zero across the entire range, which is the signature you are looking
# for.

# %% [markdown]
# ## The corner nobody checks
#
# A **reference-free** judge, one shown a rubric and never a gold answer, cannot
# fail either test. There is no reference to corrupt.
#
# Running the test there produces a comfortable zero that means nothing at all.
# The tool refuses a verdict instead:

# %%
r = meta_evaluate(rows, blind_judge, mode="swap", name="rubric-only", reference_free=True)
print(table([r]))

# %% [markdown]
# **A meta-evaluation that cannot fail is exactly the thing to be suspicious
# of.** If your judge is reference-free, this test is not the check you need.
# Validate it against human labels instead, and say so in the report rather than
# reporting a zero.

# %% [markdown]
# ## Your turn
#
# **Wrap your own judge.** A judge here is any function taking rows and
# returning `"pass"` or `"fail"` per row. Ten lines around an API call is enough.

# %%
def my_judge(rows):
    """Replace this. Rules-based, an API call, whatever you actually use."""
    return ["pass" if r["output"].strip().endswith(r["reference"].strip()[-4:])
            else "fail" for r in rows]


print(table([
    meta_evaluate(rows, my_judge, mode="flip", frac=0.20, name="mine"),
    meta_evaluate(rows, my_judge, mode="swap", frac=0.20, name="mine"),
]))

# %% [markdown]
# Then swap in your own golden set. The format is one JSON object per line with
# `id`, `output`, `reference`, `label`. Twenty rows is enough to get a signal
# and takes about twenty minutes to write.

# %% [markdown]
# ## What to take away
#
# 1. **A judge that reads nothing can score well** on a mostly-pass golden set,
#    which is every golden set anyone curates.
# 2. **The standard flip test can pass a blind judge by luck.** Whether it does
#    depends on which rows the sampler picked. Report the seed, or run several.
# 3. **`swap` cannot be fooled that way.** Permuting something the judge never
#    reads changes nothing it says, so zero means zero.
# 4. **A reference-free judge cannot fail either test.** That is not a pass.
# 5. **Nobody runs this.** It takes an afternoon and it changes how you read
#    every evaluation number you will ever be shown, including your own.
#
# ---
#
# Part of *Agent Engineering*. Previous: **01 · Evaluating Agent Work**.
