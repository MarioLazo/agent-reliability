"""Tests for the voice control-logic models.

Each of the four findings the module teaches is asserted here, so that
"improving" a fixture cannot quietly delete a lesson.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agenteng.voice import (DEFAULT_PIPELINE, DETECTION, BargeInPolicy, CallState,
                            LatencyBudget, cab_noise, evaluate_bargein, from_utterance_only,
                            margin, readback_residual, top, with_state)


class TestLatency(unittest.TestCase):
    def test_an_ordinary_pipeline_is_already_slow(self):
        b = LatencyBudget()
        self.assertGreater(b.total("p50"), 600)
        self.assertIn(b.feel("p50"), ("slow", "broken: callers will talk over it"))

    def test_p95_is_worse_than_p50_and_that_is_what_callers_remember(self):
        b = LatencyBudget()
        self.assertGreater(b.total("p95"), b.total("p50"))
        self.assertEqual(b.feel("p95"), "broken: callers will talk over it")

    def test_the_model_is_a_minority_of_the_budget(self):
        """The finding: optimising the model budgets a third of the problem."""
        b = LatencyBudget()
        share = dict((n, f) for n, _, f in b.share("p50"))
        self.assertLess(share["model first token"], 0.5)

    def test_removing_the_model_entirely_is_still_slow(self):
        b = LatencyBudget([s for s in DEFAULT_PIPELINE if s.name != "model first token"])
        self.assertGreater(b.total("p50"), 500)


class TestBargeIn(unittest.TestCase):
    def setUp(self):
        self.events = cab_noise()

    def test_the_cab_is_deliberately_hostile(self):
        self.assertGreater(len(self.events), 100)
        self.assertTrue(any(e.kind == "speech" for e in self.events))
        self.assertTrue(any(e.kind == "noise" and e.classified_voice for e in self.events),
                        "the radio is the nasty case: loud and voice-like")

    def test_no_policy_hits_the_two_percent_target_without_missing_interruptions(self):
        """The finding. Three knobs do not tame a truck cab.

        If a future tuning makes this pass, the fixture got easier, not the
        problem. Check the fixture before celebrating.
        """
        for pol in (BargeInPolicy(-50, False, 100), BargeInPolicy(-40, True, 250),
                    BargeInPolicy(-30, True, 500)):
            r = evaluate_bargein(self.events, pol)
            acceptable = r["false_barge_in_rate"] < 0.02 and r["missed_interruption_rate"] < 0.05
            self.assertFalse(acceptable, f"unexpectedly acceptable: {pol}")

    def test_tightening_trades_false_positives_for_missed_interruptions(self):
        loose = evaluate_bargein(self.events, BargeInPolicy(-50, False, 100))
        tight = evaluate_bargein(self.events, BargeInPolicy(-30, True, 500))
        self.assertLess(tight["false_barge_in_rate"], loose["false_barge_in_rate"])
        self.assertGreater(tight["missed_interruption_rate"], loose["missed_interruption_rate"])


class TestIntent(unittest.TestCase):
    UTTERANCE = "I'm not going to make it"

    def test_the_utterance_alone_does_not_separate_the_intents(self):
        """The finding: a four-way tie. No classifier recovers absent information."""
        self.assertLess(margin(from_utterance_only(self.UTTERANCE)), 0.05)

    def test_state_resolves_it(self):
        cases = [
            (CallState(180, 6.0, 40, True, 0), "mechanical_breakdown"),
            (CallState(200, 0.4, 30, False, 0), "out_of_hours_service"),
            (CallState(240, 5.0, 5, False, 3), "cannot_reach_consignee"),
            (CallState(20, 6.0, 60, False, 0), "late_for_delivery_window"),
        ]
        for state, expected in cases:
            d = with_state(self.UTTERANCE, state)
            with self.subTest(expected=expected):
                self.assertEqual(top(d)[0], expected)
                self.assertGreater(margin(d), 0.3)

    def test_distributions_are_normalised(self):
        for d in (from_utterance_only(self.UTTERANCE),
                  with_state(self.UTTERANCE, CallState(180, 6.0, 40, True, 0))):
            self.assertAlmostEqual(sum(d.values()), 1.0, places=6)


class TestReadback(unittest.TestCase):
    def test_better_detection_lets_fewer_errors_through(self):
        rates = [readback_residual(20_000, detection=d)["escaped"]
                 for d in (0.5, 0.63, 0.9)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_even_the_best_environment_leaks_over_a_year(self):
        """The finding: readback is a very good control and not closure."""
        r = readback_residual(500 * 250, detection=DETECTION["en_route"])
        self.assertGreater(r["escaped"], 0)

    def test_caught_plus_escaped_accounts_for_every_error(self):
        r = readback_residual(5_000)
        self.assertEqual(r["caught"] + r["escaped"], r["introduced"])

    def test_it_is_seeded_and_repeatable(self):
        self.assertEqual(readback_residual(5_000), readback_residual(5_000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
