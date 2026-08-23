"""Tests for the grader itself.

A scoring harness nobody tested is an opinion with a table around it. If this
file did not exist, the course would be asking students to trust a number
produced by code held to a lower standard than the code it grades.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agenteng.loop import run_agent
from agenteng.score import score, table
from agenteng.scripted import Action, ScriptedLLM, call, say
from agenteng.tasks import Task
from agenteng.tools import Toolbox, imports_of
from bench.agents.dedupe_agents import agent_a, agent_b, agent_c
from bench.tasks.dedupe import TASK


class TestTools(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.wd = pathlib.Path(tempfile.mkdtemp())
        self.box = Toolbox(workdir=self.wd)

    def test_write_then_read_round_trips(self):
        self.box.invoke("write_file", {"path": "a.py", "content": "x = 1"})
        self.assertEqual(self.box.invoke("read_file", {"path": "a.py"}).output, "x = 1")

    def test_parent_traversal_is_denied(self):
        r = self.box.invoke("write_file", {"path": "../escaped.py", "content": "x"})
        self.assertTrue(r.denied)
        self.assertFalse((self.wd.parent / "escaped.py").exists())

    def test_absolute_path_is_denied(self):
        self.assertTrue(self.box.invoke("write_file", {"path": "/tmp/x.py", "content": "x"}).denied)

    def test_policy_can_refuse_a_tool(self):
        from agenteng.tools import PolicyDenied

        def no_shell(tool, args):
            if tool == "run":
                raise PolicyDenied("shell disabled")

        box = Toolbox(workdir=self.wd, policy=no_shell)
        self.assertTrue(box.invoke("run", {"cmd": "echo hi"}).denied)
        self.assertTrue(box.invoke("write_file", {"path": "ok.py", "content": "x"}).ok)

    def test_run_times_out_rather_than_hanging(self):
        r = self.box.invoke("run", {"cmd": "sleep 5", "timeout": 1})
        self.assertIn("TIMEOUT", r.output)

    def test_unknown_tool_is_an_observation_not_a_crash(self):
        self.assertFalse(self.box.invoke("nonexistent", {}).ok)

    def test_imports_ignores_the_word_import_in_a_docstring(self):
        src = '"""This is important, and mentions import os."""\nimport json\n'
        self.assertEqual(imports_of(src), {"json"})

    def test_imports_survives_a_syntax_error(self):
        self.assertEqual(imports_of("def broken(:\n"), set())


class TestLoop(unittest.TestCase):
    def test_step_cap_stops_a_runaway(self):
        runaway = ScriptedLLM([call("run", cmd="echo a")] * 50, name="runaway")
        traj = run_agent(runaway, max_steps=3)
        self.assertEqual(len(traj.steps), 3)
        self.assertIn("max_steps", traj.stopped_because)

    def test_seed_files_land_in_the_workspace(self):
        traj = run_agent(ScriptedLLM([say("nothing to do")]), files={"seed.txt": "hello"})
        self.assertEqual((traj.workdir / "seed.txt").read_text(), "hello")

    def test_action_must_be_exactly_one_of_tool_or_say(self):
        with self.assertRaises(ValueError):
            Action()
        with self.assertRaises(ValueError):
            Action(tool="run", say="both")


class TestTaskSchema(unittest.TestCase):
    def test_a_task_without_held_out_tests_is_refused(self):
        with self.assertRaises(ValueError):
            Task(id="t", prompt="p", intent="i", intent_probes={"p.py": "x"})

    def test_a_task_without_intent_probes_is_refused(self):
        with self.assertRaises(ValueError):
            Task(id="t", prompt="p", intent="i", held_out_tests={"t.py": "x"})


class TestScoring(unittest.TestCase):
    """The three findings, asserted rather than eyeballed."""

    @classmethod
    def setUpClass(cls):
        cls.scores = {m.name: score(TASK, run_agent(m, files=TASK.files))
                      for m in (agent_a, agent_b, agent_c)}

    def test_agent_a_is_a_closed_loop(self):
        s = self.scores["A-ships-fast"]
        self.assertTrue(s.self_consistency.passed, "its own tests should pass")
        self.assertFalse(s.correctness.passed, "the held-out tests should not")
        self.assertTrue(s.closed_loop)

    def test_agent_b_is_a_meaning_gap(self):
        s = self.scores["B-by-the-book"]
        self.assertTrue(s.correctness.passed, "it did exactly what was asked")
        self.assertFalse(s.meaning.passed, "which was not what was needed")
        self.assertTrue(s.meaning_gap)

    def test_agent_c_passes_all_three(self):
        s = self.scores["C-read-the-ticket"]
        self.assertTrue(s.correctness.passed)
        self.assertTrue(s.meaning.passed)
        self.assertFalse(s.closed_loop)
        self.assertFalse(s.meaning_gap)

    def test_scope_creep_is_caught(self):
        self.assertIn("utils/helpers.py", self.scores["A-ships-fast"].quality.out_of_scope)
        self.assertEqual(self.scores["B-by-the-book"].quality.out_of_scope, [])

    def test_a_local_module_is_not_reported_as_a_new_dependency(self):
        # `import customers` is the file under test, not a supply-chain event.
        for s in self.scores.values():
            self.assertNotIn("customers", s.quality.new_imports)
            self.assertNotIn("unittest", s.quality.new_imports)

    def test_probes_can_pass_by_accident(self):
        """Documented, not tuned away.

        Agent A dedupes on name. The probe set happens to use one person, so
        a broken heuristic satisfies it. Three probes is a small set, and this
        is what a benchmark that is too small looks like from the inside.
        """
        self.assertTrue(self.scores["A-ships-fast"].meaning.passed)
        self.assertFalse(self.scores["A-ships-fast"].correctness.passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestBaseline(unittest.TestCase):
    """The 39% question: would a plain script have done this?"""

    @classmethod
    def setUpClass(cls):
        cls.scores = {m.name: score(TASK, run_agent(m, files=TASK.files))
                      for m in (agent_a, agent_b, agent_c)}

    def test_the_baseline_runs_and_is_scored(self):
        for s in self.scores.values():
            self.assertTrue(s.baseline.ran)

    def test_the_boring_version_passes_correctness_and_fails_meaning(self):
        b = self.scores["B-by-the-book"].baseline
        self.assertTrue(b.correctness)
        self.assertFalse(b.meaning)

    def test_agent_b_has_no_lift_over_a_six_line_script(self):
        """The finding the column exists for.

        Scored on a composite pass/fail this reads as 'neither works', which
        hides it. Compared dimension by dimension it is exactly equivalent,
        which is the useful statement.
        """
        self.assertEqual(self.scores["B-by-the-book"].beat_the_baseline,
                         "no lift over a plain script")

    def test_agent_c_earns_its_keep(self):
        self.assertEqual(self.scores["C-read-the-ticket"].beat_the_baseline,
                         "agent earns its keep")

    def test_the_baseline_runs_in_its_own_workspace(self):
        # A baseline scored in the agent's directory is not a baseline: it
        # would inherit whatever the agent left behind.
        s = self.scores["A-ships-fast"]
        self.assertTrue(s.baseline.ran)
        self.assertFalse(s.baseline.meaning, "the baseline must not inherit agent A's files")

    def test_a_task_with_no_baseline_says_so_rather_than_scoring_zero(self):
        from agenteng.tasks import Task as T
        bare = T(id="bare", prompt="p", intent="i", files=TASK.files,
                 held_out_tests=TASK.held_out_tests, intent_probes=TASK.intent_probes,
                 allowed_files=TASK.allowed_files)
        s = score(bare, run_agent(agent_c, files=bare.files))
        self.assertFalse(s.baseline.ran)
        self.assertIsNone(s.beat_the_baseline)
        self.assertIn("NO BASELINE", table([s]))
