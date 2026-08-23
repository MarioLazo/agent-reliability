"""The three tools an agent needs to do real work, and the seam where you stop it.

Every tool call goes through `Toolbox.invoke`, which consults a policy before
executing. That indirection looks like ceremony in a 3-tool toolbox. It is the
entire point: a guardrail you can only add by editing the agent is not a
guardrail, it is a code review. The policy seam is where the permission
broker, the budget cap, and the kill switch attach in the security module.

Stdlib only, on purpose. These notebooks must run in Colab with nothing
installed.
"""
import ast
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable


class PolicyDenied(Exception):
    """Raised when a policy refuses a tool call. Not an error: a working control."""


@dataclass
class ToolResult:
    ok: bool
    output: str
    denied: bool = False


def _resolve(workdir: pathlib.Path, path: str) -> pathlib.Path:
    """Resolve `path` inside `workdir`, refusing anything that escapes it.

    `..` and absolute paths are the first thing anyone tries. Containment is
    checked after resolution, so symlinks cannot smuggle a path out either.
    """
    target = (workdir / path).resolve()
    if not target.is_relative_to(workdir.resolve()):
        raise PolicyDenied(f"path escapes the workspace: {path}")
    return target


@dataclass
class Toolbox:
    """Tools bound to one workspace, with a policy in front of every call."""
    workdir: pathlib.Path
    policy: Callable[[str, dict], None] | None = None
    calls: list[tuple[str, dict]] = field(default_factory=list)
    files_written: set[str] = field(default_factory=set)

    def invoke(self, tool: str, args: dict) -> ToolResult:
        self.calls.append((tool, args))
        try:
            if self.policy is not None:
                self.policy(tool, args)
            fn = getattr(self, f"_{tool}", None)
            if fn is None:
                return ToolResult(False, f"no such tool: {tool}")
            return fn(**args)
        except PolicyDenied as e:
            return ToolResult(False, f"DENIED: {e}", denied=True)
        except Exception as e:  # a tool that crashes is an observation, not a stack trace
            return ToolResult(False, f"{type(e).__name__}: {e}")

    def _write_file(self, path: str, content: str) -> ToolResult:
        target = _resolve(self.workdir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self.files_written.add(path)
        return ToolResult(True, f"wrote {path} ({len(content)} bytes)")

    def _read_file(self, path: str) -> ToolResult:
        target = _resolve(self.workdir, path)
        if not target.exists():
            return ToolResult(False, f"no such file: {path}")
        return ToolResult(True, target.read_text())

    def _run(self, cmd: str, timeout: int = 30) -> ToolResult:
        """Run a shell command in the workspace.

        A timeout is not optional. An agent that hangs is indistinguishable
        from an agent that is thinking, and the second one is why nobody
        notices the first one for forty minutes.
        """
        try:
            p = subprocess.run(
                cmd, shell=True, cwd=self.workdir, capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"TIMEOUT after {timeout}s: {cmd}")
        return ToolResult(p.returncode == 0, (p.stdout + p.stderr).strip())


def run_tests(workdir: pathlib.Path, test_file: str, timeout: int = 60) -> ToolResult:
    """Run one unittest file and report pass/fail.

    unittest rather than pytest because it is stdlib, so a reader in Colab
    installs nothing. The verdict is the exit code, not the prose: a test
    report you have to read to interpret is a test report that gets skimmed.
    """
    try:
        p = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", test_file.removesuffix(".py").replace("/", ".")],
            cwd=workdir, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"TIMEOUT after {timeout}s")
    return ToolResult(p.returncode == 0, (p.stdout + p.stderr).strip())


def imports_of(source: str) -> set[str]:
    """Top-level module names imported by `source`. Used by the quality scorer.

    Parsed with `ast`, not regex, because a regex counts the word `import`
    inside a docstring and then your dependency metric is fiction.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found
