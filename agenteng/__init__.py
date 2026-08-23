"""agenteng: the runnable harness behind the Agent Engineering course.

Everything here is stdlib-only and deterministic by default, so a lesson
reproduces on any machine with no API key and no install.

    from agenteng import ScriptedLLM, call, say, run_agent, score, table

The three questions the scorer keeps separate, because they fail separately:

    correctness   did it do what I asked?         held-out tests
    quality       did it do it well?              scope, dependencies, size
    meaning       did I ask for the right thing?  intent probes
"""
from .loop import Step, Trajectory, run_agent
from .score import Quality, Score, Verdict, score, table
from .scripted import Action, ScriptedLLM, SuggestibleLLM, call, say
from .tasks import Task
from .tools import PolicyDenied, ToolResult, Toolbox, imports_of, run_tests

__all__ = [
    "Action", "PolicyDenied", "Quality", "ScriptedLLM", "Score", "Step",
    "SuggestibleLLM", "Task", "Toolbox", "ToolResult", "Trajectory", "Verdict",
    "call", "imports_of", "run_agent", "run_tests", "say", "score", "table",
]

__version__ = "0.1.0"
