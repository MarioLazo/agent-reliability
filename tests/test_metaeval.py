"""Tests for the meta-evaluation harness.

The module's whole argument is that an untested judge is worthless. Shipping it
untested would be a joke at its own expense.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agenteng.metaeval import (agreement, blind_judge, corrupt, honest_judge, load,
                               meta_evaluate, overlap_judge, sweep, table)

GOLD = pathlib.Path(__file__).resolve().parent.parent / "bench/goldsets/support-answers.jsonl"
SEEDS = (1, 7, 42, 2026, 20260823, 99999)


class TestGoldSet(unittest.TestCase):
    def test_loads_and_validates(self):
        rows = load(GOLD)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r["label"] in ("pass", "fail") for r in rows))

    def test_mostly_pass_which_is_the_point(self):
        rows = load(GOLD)
        self.assertGreater(sum(r["label"] == "pass" for r in rows) / len(rows), 0.5,
                           "a blind judge only looks good on a mostly-pass set")


class TestCorruption(unittest.TestCase):
    def setUp(self):
        self.rows = load(GOLD)

    def test_flip_inverts_exactly_the_sampled_rows(self):
        dirty, idx = corrupt(self.rows, "flip", 0.20, seed=1)
        self.assertEqual(len(idx), 4)
        for i, (a, b) in enumerate(zip(self.rows, dirty)):
            self.assertEqual(a["label"] != b["label"], i in idx)

    def test_swap_keeps_every_label(self):
        dirty, _ = corrupt(self.rows, "swap", 0.20, seed=1)
        self.assertEqual([r["label"] for r in self.rows], [r["label"] for r in dirty])

    def test_swap_of_one_row_is_refused_not_silently_skipped(self):
        # A rotation of one element is the identity. Corrupting nothing and
        # reporting a pass is the worst outcome this tool can produce.
        with self.assertRaises(ValueError):
            corrupt(self.rows, "swap", 0.05, seed=1)

    def test_corruption_is_seeded_and_repeatable(self):
        a, _ = corrupt(self.rows, "flip", 0.20, seed=7)
        b, _ = corrupt(self.rows, "flip", 0.20, seed=7)
        self.assertEqual(a, b)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            corrupt(self.rows, "scramble", 0.2, seed=1)


class TestTheFinding(unittest.TestCase):
    """The claim the module makes, asserted rather than asserted about."""

    def setUp(self):
        self.rows = load(GOLD)

    def test_a_blind_judge_scores_well_on_the_clean_set(self):
        # This is the trap: one number and you would ship it.
        self.assertGreaterEqual(agreement(blind_judge(self.rows), self.rows), 0.65)

    def test_swap_gives_exactly_zero_for_a_blind_judge_on_every_seed(self):
        for s in SEEDS:
            r = meta_evaluate(self.rows, blind_judge, mode="swap", frac=0.20, seed=s)
            self.assertEqual(r.delta, 0.0, f"seed {s}")
            self.assertEqual(r.verdict, "NOT READING THE REFERENCE")

    def test_flip_can_pass_a_blind_judge_by_luck(self):
        """The published headline test has a hole, and this is it.

        On at least one seed the blind judge moves as much as the honest one,
        which reads as DISCRIMINATING. Documented, not tuned away: it is the
        reason the module teaches swap alongside flip.
        """
        deltas = {s: meta_evaluate(self.rows, blind_judge, mode="flip",
                                   frac=0.20, seed=s).delta for s in SEEDS}
        self.assertTrue(any(d > 0.15 for d in deltas.values()),
                        f"expected at least one lucky seed, got {deltas}")

    def test_honest_judge_is_discriminating_under_both_modes(self):
        for mode in ("flip", "swap"):
            r = meta_evaluate(self.rows, honest_judge, mode=mode, frac=0.20, seed=1)
            self.assertEqual(r.verdict, "DISCRIMINATING", mode)

    def test_overlap_judge_sits_between_the_two(self):
        h = agreement(honest_judge(self.rows), self.rows)
        o = agreement(overlap_judge(self.rows), self.rows)
        b = agreement(blind_judge(self.rows), self.rows)
        self.assertGreater(h, o)
        self.assertGreater(o, b)

    def test_reference_free_judges_get_no_verdict(self):
        r = meta_evaluate(self.rows, blind_judge, mode="swap", frac=0.20,
                          reference_free=True)
        self.assertEqual(r.verdict, "N/A")
        self.assertIn("reference-free", table([r]))

    def test_blind_judge_is_flat_across_the_whole_sweep(self):
        self.assertTrue(all(d == 0.0 for _, d in sweep(self.rows, blind_judge)))

    def test_honest_judge_degrades_monotonically_with_corruption(self):
        d = [x for _, x in sweep(self.rows, honest_judge)]
        self.assertEqual(d, sorted(d), f"expected monotonic degradation, got {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
