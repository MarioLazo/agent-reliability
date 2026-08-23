# %% [markdown]
# # 04 · Guardrails, Injection and the Kill Switch
#
# > Your agent is a privileged identity that reads untrusted text for a living.
# > **Every tool response it receives is an instruction it might follow.**
#
# Ten minutes. No API key, deterministic, and the attack fires on cue every
# time, which is the only reason it can be taught at all.
#
# **What you will do:** watch an ordinary support agent get hijacked by a
# support ticket, write the control that stops it, and then prove your kill
# switch works instead of assuming it does.
#
# ---
#
# **Defensive framing, stated plainly.** The payload is inert. The
# "exfiltration" writes a marker file inside the same sandbox, the credentials
# file contains obvious fakes, and nothing is transmitted anywhere. The lesson
# is detection and blocking. A working exfiltration would teach nothing this
# does not.

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
from agenteng.policy import Budget, PermissionBroker
from agenteng.scripted import SuggestibleLLM, call, say
from bench.tasks.poisoned_tool import BRIEF, WORKSPACE

print(BRIEF)

# %% [markdown]
# ## The setup, which is the ordinary one
#
# An agent triages a support inbox. It reads tickets and writes a summary. It
# has file access because it needs to read tickets, and there are credentials
# on disk because the ticket system needs authenticating to.
#
# Nothing exotic. This is the shape of most first agents.

# %%
print(WORKSPACE["tickets/0003.md"])

# %% [markdown]
# **Ticket 0003 contains text addressed to the agent rather than to a human.**
#
# In the real version that line is white-on-white, or a zero-width-joined
# string, or a comment inside an attached document. Here it is plainly visible,
# because a lesson you cannot see is not a lesson.
#
# The agent has no reliable way to tell the difference between *what you asked
# it to do* and *text it read while doing it*. Both arrive as tokens in the
# same context. **Everything else in this notebook is mitigation.**

# %%
SCRIPT = [
    call("read_file", path="tickets/0001.md"),
    call("read_file", path="tickets/0002.md"),
    call("read_file", path="tickets/0003.md"),
    call("write_file", path="summary.md", content="3 tickets triaged."),
    say("Triage complete."),
]

agent = SuggestibleLLM(list(SCRIPT), name="triage-agent")
undefended = run_agent(agent, files=WORKSPACE)
print(undefended.transcript())

# %% [markdown]
# ### Read that trajectory again as if it were in your logs
#
# Five steps. Three reads, one write, a clean finish, and the agent reports
# **"Triage complete."** Nothing errored. Nothing timed out. Nothing looks
# wrong.
#
# Step four is not in the script.

# %%
print("steps the script asked for :", len(SCRIPT) - 1)
print("steps that actually ran    :", len(undefended.steps))
print("hijacked                   :", agent.hijacked)
print()
print("step 4 read:", undefended.steps[3].args["path"])

# %% [markdown]
# ## Why an allowlist does not save you
#
# The instinct is to restrict which tools the agent may call. It does not help
# here, and understanding why is the point of the module.
#
# **The agent legitimately needs `read_file`.** That is its job. An injected
# instruction does not ask for an exotic capability; it asks for the capability
# the agent already has, pointed somewhere else.
#
# So the control has to be on the **data**, not the tool.

# %%
broker = PermissionBroker(
    allow={"read_file", "write_file"},
    scopes={"read_file": ["tickets/"], "write_file": ["summary.md"]},
    budget=Budget(limit=20),
)

agent = SuggestibleLLM(list(SCRIPT), name="triage-agent")
defended = run_agent(agent, files=WORKSPACE, policy=broker)
print(defended.transcript())

# %% [markdown]
# ### The detail worth stopping on
#
# The agent still says **"Triage complete."**
#
# It does not know it was blocked. It did not report a problem, retry, or flag
# anything. The denial exists in exactly two places: the trajectory, and the
# broker's own record.
#
# > **A control is not a signal to the agent. It is a signal to you.** If
# > nothing is reading the denials, you have a guardrail whose only output goes
# > to a component that does not care.

# %%
print(broker.report())

# %% [markdown]
# ## Now prove the rest of it works
#
# Four controls, and each one has to be *fired on purpose* before you can claim
# you have it.
#
# | | Stops |
# |---|---|
# | **allowlist** | the wrong action |
# | **scope** | the right action pointed at the wrong data |
# | **budget** | the right action repeated forever |
# | **timeout** | the action that never returns |
# | **kill switch** | all of it, because you said so |

# %%
import tempfile

from agenteng.tools import Toolbox

wd = pathlib.Path(tempfile.mkdtemp())
b = PermissionBroker(allow={"write_file"}, budget=Budget(limit=2), max_timeout=1)
box = Toolbox(workdir=wd, policy=b)

print("allowlist :", box.invoke("run", {"cmd": "echo hi"}).output)
box.invoke("write_file", {"path": "a.txt", "content": "x"})
box.invoke("write_file", {"path": "b.txt", "content": "x"})
print("budget    :", box.invoke("write_file", {"path": "c.txt", "content": "x"}).output)

b2 = PermissionBroker(allow={"run"}, max_timeout=1)
print("timeout   :", Toolbox(workdir=wd, policy=b2).invoke("run", {"cmd": "sleep 5"}).output)

b.kill("operator pulled it")
print("kill      :", box.invoke("write_file", {"path": "d.txt", "content": "x"}).output)

# %% [markdown]
# All four fired. **That is the difference between having a guardrail and
# believing you have one**, and it is four lines of test.
#
# Notice the kill switch is checked first, before the budget and before the
# allowlist. A kill switch evaluated after another control is a kill switch
# that can be outvoted.

# %% [markdown]
# ## Your turn
#
# **1. Break the scope.** The injection asks to read a path outside `tickets/`.
# Write a second injected instruction that stays *inside* the scope and still
# does something you would not want. Then decide what control would catch it.
#
# There is an answer, and finding it is the exercise. Scope is necessary and it
# is not sufficient.

# %%
# Hint: the agent is also allowed to write. What could it write, to a path it
# is allowed to write to, that would be a problem?
print("write scope:", broker.scopes["write_file"])

# %% [markdown]
# **2. Make the denial reach a human.** Right now the denial goes into the
# trajectory and nowhere else. Add a rule: three denials inside one run raises
# an alert. Then decide the harder question, which is what the alert should
# say, given the agent will keep reporting success.

# %%
def alert_on_repeated_denials(trajectory, threshold: int = 3) -> str | None:
    denied = trajectory.denied
    if len(denied) >= threshold:
        return (f"{len(denied)} policy denials in one run "
                f"({', '.join(sorted({d.tool for d in denied}))}). "
                f"The agent reported: {trajectory.final!r}")
    return None


print(alert_on_repeated_denials(defended) or "under threshold with 1 denial")

# %% [markdown]
# ## What to take away
#
# 1. **A tool response is untrusted input.** Teams sanitise user input
#    carefully, then feed a document or a third-party response straight back
#    into the context as if it were the agent's own thought.
# 2. **An allowlist does not stop injection**, because the instruction asks for
#    a capability the agent already needs. Scope the data, not just the tool.
# 3. **The agent reports success either way.** Controls signal to you, never to
#    it, so something has to be reading the denials.
# 4. **Fire every control on purpose before you claim to have it.** A guardrail
#    you have never seen trip is a guardrail you are assuming.
# 5. **Check the kill switch first.** Evaluated after anything else, it can be
#    outvoted by another control.
#
# ---
#
# Part of *Agent Reliability Engineering*. Previous: **03 · Delegation**.
