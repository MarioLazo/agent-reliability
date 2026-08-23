"""Tests for the harness-effect demonstration.

The module's claim is that a harness moves the score without the model
changing. That claim is asserted here, along with the mechanism, so that a
future tidy-up cannot quietly delete either.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agenteng.loop import run_agent
from agenteng.score import score
from bench.agents.dedupe_agents import WITH_REPAIR, agent_d
from bench.harnesses import HARNESSES
from bench.tasks.dedupe import TASK


def rate(h) -> float:
    ok = 0
    for m in WITH_REPAIR:
        s = score(TASK, run_agent(m, files=TASK.files, policy=h.policy(), max_steps=h.max_steps))
        ok += s.correctness.passed and s.meaning.passed
    return ok / len(WITH_REPAIR)


class TestHarnessEffect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rates = {h.name: rate(h) for h in HARNESSES}

    def test_the_score_moves_across_harnesses(self):
        """The claim. Same agents, same task, same fixture."""
        self.assertGreater(max(self.rates.values()) - min(self.rates.values()), 0.2)

    def test_the_generous_harness_is_not_the_only_good_one(self):
        # If only the unrestricted config worked, the lesson would be "do not
        # restrict", which is the wrong lesson.
        self.assertEqual(self.rates["generous"], self.rates["standard"])

    def test_every_failing_harness_is_a_defensible_setting(self):
        for name in ("thrifty", "short-leash", "locked-down"):
            self.assertLess(self.rates[name], self.rates["generous"], name)


class TestMechanism(unittest.TestCase):
    """The harness moves the score by deciding whether self-correction closes."""

    def _run(self, h):
        t = run_agent(agent_d, files=TASK.files, policy=h.policy(), max_steps=h.max_steps)
        return t, score(TASK, t), agent_d.repaired

    def test_the_agent_repairs_when_the_harness_permits_it(self):
        for name in ("generous", "standard"):
            h = next(x for x in HARNESSES if x.name == name)
            _, s, repaired = self._run(h)
            self.assertTrue(repaired, name)
            self.assertTrue(s.meaning.passed, name)

    def test_a_step_cap_below_the_loop_length_prevents_the_repair(self):
        h = next(x for x in HARNESSES if x.name == "short-leash")
        t, s, repaired = self._run(h)
        self.assertFalse(repaired)
        self.assertEqual(len(t.steps), h.max_steps)
        self.assertFalse(s.meaning.passed)

    def test_removing_the_shell_prevents_the_agent_finding_its_own_bug(self):
        h = next(x for x in HARNESSES if x.name == "locked-down")
        _, s, repaired = self._run(h)
        self.assertFalse(repaired)
        self.assertFalse(s.meaning.passed)

    def test_a_budget_that_runs_out_prevents_the_repair(self):
        h = next(x for x in HARNESSES if x.name == "thrifty")
        _, s, repaired = self._run(h)
        self.assertFalse(s.meaning.passed)

    def test_scripted_agents_alone_show_no_spread(self):
        """The negative result that produced the design.

        Agents that do not react cannot be affected by a harness. This is
        asserted so nobody 'fixes' the demonstration by reverting to them.
        """
        from bench.agents.dedupe_agents import ALL
        rates = []
        for h in HARNESSES:
            ok = sum(1 for m in ALL
                     if (lambda s: s.correctness.passed and s.meaning.passed)(
                         score(TASK, run_agent(m, files=TASK.files, policy=h.policy(),
                                               max_steps=h.max_steps))))
            rates.append(ok)
        self.assertEqual(len(set(rates)), 1, "non-reactive agents should show zero spread")


if __name__ == "__main__":
    unittest.main(verbosity=2)
