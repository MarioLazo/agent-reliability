"""Tests for the topology model and the cascade fixture.

The module's argument is arithmetic, so the arithmetic gets tested. A course
that tells students to verify analytic results by simulation and then ships
unsimulated formulas would be making its own point against itself.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agenteng.reliability import (TOPOLOGIES, _chain, _fanout_verify, better_agents_vs_verifier,
                                  compare, monte_carlo)
from agenteng.tools import run_tests
from bench.agents.pipeline_agents import HELD_OUT, PIPELINE, run_pipeline


class TestArithmetic(unittest.TestCase):
    def test_three_stages_at_ninety_percent_is_seventy_three(self):
        self.assertAlmostEqual(_chain(0.9), 0.729, places=3)

    def test_every_analytic_formula_matches_its_simulation(self):
        for t in TOPOLOGIES:
            with self.subTest(topology=t.name):
                self.assertAlmostEqual(t.reliability(0.9), monte_carlo(t.name, 0.9, trials=40_000),
                                       delta=0.01)

    def test_reliability_falls_as_stages_are_added(self):
        vals = [_chain(0.9, n) for n in (1, 2, 3, 4, 5)]
        self.assertEqual(vals, sorted(vals, reverse=True))


class TestTheFinding(unittest.TestCase):
    def test_a_partial_verifier_beats_making_every_stage_better(self):
        d = better_agents_vs_verifier(p=0.9, bump=0.05, recall=0.8)
        self.assertGreater(d["add_verifier"], d["better_agents"])
        self.assertGreater(d["better_agents"], d["baseline"])

    def test_the_verifier_never_wins_on_cost_efficiency(self):
        """Documented, not tuned away.

        Even a perfect verifier loses to the plain chain on
        reliability-per-unit-cost. The notebook says so, because
        cost-efficiency is the wrong objective when failure is expensive, and
        an exercise that implied a crossover would have been teaching a
        number that does not exist.
        """
        chain_eff = _chain(0.9) / 3.0
        cost = 3.0 + 3.0 + 3.0 * (1 - 0.9)
        for recall in (0.0, 0.5, 1.0):
            self.assertLess(_fanout_verify(0.9, recall=recall) / cost, chain_eff)


class TestCascade(unittest.TestCase):
    """Every agent correct, the system wrong."""

    @classmethod
    def setUpClass(cls):
        cls.wd, cls.trajs = run_pipeline()

    def test_all_three_stages_completed(self):
        self.assertEqual(len(self.trajs), len(PIPELINE))
        for t in self.trajs:
            self.assertEqual(t.stopped_because, "said-done")
            self.assertTrue(t.final)

    def test_no_agent_hit_an_error(self):
        for t in self.trajs:
            for step in t.steps:
                self.assertTrue(step.ok, f"{t.agent} step {step.n}: {step.result[:80]}")

    def test_the_pipelines_own_tests_pass(self):
        self.assertTrue(run_tests(self.wd, "test_checkout.py").ok)

    def test_the_ticket_is_not_satisfied(self):
        (self.wd / "test_reality.py").write_text(HELD_OUT)
        r = run_tests(self.wd, "test_reality.py")
        self.assertFalse(r.ok)
        self.assertIn("20.0 != 80.0", r.output)

    def test_the_error_entered_at_stage_one(self):
        # The spec, not the implementation, is the artifact that is wrong.
        self.assertIn("discount amount", (self.wd / "spec.py").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
