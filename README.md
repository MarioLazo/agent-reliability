<div align="center">

# Agent Reliability Engineering

### 🕒 The 3pm Test

**Would you deploy this agent on a Tuesday at 3pm?**

*Not "does it work." Every agent works in the room where it was built.*

[![verify](https://github.com/MarioLazo/agent-reliability/actions/workflows/verify.yml/badge.svg)](https://github.com/MarioLazo/agent-reliability/actions/workflows/verify.yml)
![tests](https://img.shields.io/badge/tests-85-brightgreen?style=flat-square)
![deps](https://img.shields.io/badge/dependencies-none-blue?style=flat-square)
![api key](https://img.shields.io/badge/API%20key-not%20required-blue?style=flat-square)

</div>

---

## Ten seconds, and the whole argument

One task. Three agents. All three pass their own tests.

```
agent             | correct | own tests | meaning | quality
-----------------------------------------------------------
A-ships-fast      |  FAIL   |   PASS    |  PASS   | scope creep
B-by-the-book     |  PASS   |   PASS    |  FAIL   | clean
C-read-the-ticket |  PASS   |   PASS    |  PASS   | clean
(baseline)        |  PASS   |    n/a    |  FAIL   | no agent, no model

  CLOSED LOOP  A: its own tests pass, the held-out tests do not.
  MEANING GAP  B: did what was asked, not what was needed.
  NO LIFT      B: scores exactly what a six-line script scores.
```

`own tests` is green for all three. **It is also the column every agent demo
shows you.**

B is the expensive one. Clean code, honest tests, would pass review, and it
does nothing a `for` loop and a `set` could not have done.

```bash
git clone https://github.com/MarioLazo/agent-reliability
cd agent-reliability
make verify        # 85 tests, 6 notebooks, under 5 seconds
```

No `pip install`. No API key. No network. If that surprises you, that is the point.

---

## Why reliability, and not capability

<table>
<tr><td width="50%" valign="top">

**38%**

Practitioners rank reliability, robustness and scalability ahead of everything
else, and deliberately trade capability for controllability to get it.

<sub>306 practitioners · [arXiv:2512.04123](https://arxiv.org/html/2512.04123v3)</sub>

</td><td width="50%" valign="top">

**22.2 points**

The spread on **one unchanged model** across nine harnesses, same benchmark.
80.2% at the top, 58.0% at the bottom.

<sub>Terminal-Bench 2.0, retrieved 2026-08-21</sub>

</td></tr>
<tr><td valign="top">

**39%**

Of production systems compare against a non-agentic baseline. 26% report no
meaningful baseline exists at all.

<sub>The largest measurement gap in the field</sub>

</td><td valign="top">

**79%**

Of measured multi-agent failures are specification or coordination, not
implementation.

<sub>MAST · 1,600+ annotated traces · κ = 0.88</sub>

</td></tr>
</table>

**Reliability is a property of what you build around the model.** That is the
half nobody teaches, and it is what this is.

---

## The notebooks

Every one runs offline, deterministically, in about ten seconds.

| | What it proves |
|---|---|
| **[01 · Evaluating Agent Work](notebooks/01_evaluation.ipynb)** | Correctness, quality and meaning fail independently, so one number hides two of them |
| **[02 · Judging the Judge](notebooks/02_judging_the_judge.ipynb)** | Corrupt your golden set and see whether the judge notices. The standard test is seed-dependent; a sharper one is not |
| **[03 · Delegation](notebooks/03_delegation.ipynb)** | Three stages at 90% is a 73% system. A verifier catching 80% beats making every stage 5% better |
| **[04 · Guardrails](notebooks/04_guardrails.ipynb)** | An allowlist does not stop injection, because the instruction asks for a tool the agent already needs |
| **[05 · Voice](notebooks/05_voice.ipynb)** | Latency is a correctness property. The model is a third of the budget |
| **[06 · Benchmarking](notebooks/06_benchmarking.ipynb)** | A leaderboard row that names a model is measuring `model × harness × substrate` |

---

## The keystone: a model that fails on cue

You cannot teach a failure taxonomy with a live frontier model. It will not
fail when you need it to. It fails intermittently, differently each run, and
never during the lecture.

```python
from agenteng import ScriptedLLM, call, say, run_agent, score

agent = ScriptedLLM([
    call("write_file", path="fix.py", content="..."),
    say("Done."),
], name="my-agent")

print(score(task, run_agent(agent, files=task.files)))
```

**The honest limit, stated everywhere it applies:** a scripted run proves your
*harness detects* a failure. It proves nothing about how often a real model
*produces* one. Those are different claims needing different instruments.

---

## Four rules that shaped every decision

**1 · Deterministic and offline.** A failure that fires on cue is a failure you
can teach. See above for what that does not buy you.

**2 · Stdlib only.** `unittest` not pytest, `ast` not a parser library. A reader
who has to install something is a reader who stops. The CI has no install step,
deliberately: a workflow needing a dependency would disprove the claim.

**3 · Notebooks are generated.** Source of truth is percent-format Python in
`notebooks/src/`. A committed `.ipynb` produces diffs nobody reviews, in a repo
about review discipline. `make verify` fails on a stale notebook.

**4 · Findings are asserted, not described.** Every claim in every notebook has
a test. Including the awkward ones: that the standard meta-evaluation test can
pass a blind judge by luck, that a verifier never wins on cost-efficiency, and
that a probe set can pass by accident. Tuning those away would have been the
easier commit.

---

## Status, stated plainly

**This is the runnable half of Part 2 of a three-part curriculum, and it is in
development.** Six notebooks work. The full course is 17 modules; 7 have code.

- **Part 1 · [From Vibe Coding to Agent Engineering](https://github.com/MarioLazo/vibe-coding-to-agent-engineering)**: the introduction. Free, public, and it ends where most courses end: you have built an agent and it does the thing.
- **Part 2 · this**: everything after it works.
- **Part 3 · Voice, Multimodal and Complex Intent**: in development. `05_voice` is the first of it.

---

## License

| | |
|---|---|
| **Code** (`agenteng/`, `bench/`, `tests/`, `tools/`) | MIT |
| **Content** (notebooks, prose) | CC BY 4.0 |

> "Agent Reliability Engineering" by Mario Lazo, licensed under CC BY 4.0.
