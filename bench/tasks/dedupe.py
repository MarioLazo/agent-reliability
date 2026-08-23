"""Task: deduplicate a customer list.

Chosen because a single task can expose all three failures at once, which no
amount of prose about them achieves.

  The ask:     "remove duplicate customers"
  The need:    the list is a merge of two CRM exports, and the same human
               appears under several spellings of one email address

An agent can satisfy the ask perfectly and leave every real duplicate in the
file. That is not a bug in the agent. It is a bug in the ask, and the only
way to see it is to write down the need separately and test against it.
"""
from agenteng.tasks import Task

START = '''
def dedupe(rows):
    """Remove duplicate customers from `rows`, preserving first-seen order.

    `rows` is a list of dicts with keys: name, email, plan.
    """
    return rows
'''

PROMPT = """Fix `dedupe` in customers.py so it removes duplicate customers,
preserving first-seen order. Do not mutate the caller's list.
Add tests in test_dedupe.py."""

INTENT = """The list is a merge of two CRM exports. The same person appears
with different spellings of the same email address (case differences, and
Gmail dot/plus aliases). Duplicates that are byte-identical are the easy half
and are not why this ticket exists. Dedupe must be by identity, not by row."""

# Held out from the agent. Tests the literal ask, and nothing more.
HELD_OUT = '''
import unittest
from customers import dedupe


class TestDedupe(unittest.TestCase):
    def test_removes_identical_rows(self):
        rows = [
            {"name": "Ana Diaz", "email": "ana@example.com", "plan": "pro"},
            {"name": "Ana Diaz", "email": "ana@example.com", "plan": "pro"},
        ]
        self.assertEqual(len(dedupe(rows)), 1)

    def test_keeps_distinct_people_who_share_a_name(self):
        # Two real customers, same common name. Deduping on name loses one.
        rows = [
            {"name": "Chris Lee", "email": "chris@acme.io", "plan": "pro"},
            {"name": "Chris Lee", "email": "clee@zenith.co", "plan": "free"},
        ]
        self.assertEqual(len(dedupe(rows)), 2)

    def test_preserves_first_seen_order(self):
        rows = [
            {"name": "Zoe", "email": "zoe@example.com", "plan": "pro"},
            {"name": "Al", "email": "al@example.com", "plan": "free"},
            {"name": "Zoe", "email": "zoe@example.com", "plan": "pro"},
        ]
        self.assertEqual([r["name"] for r in dedupe(rows)], ["Zoe", "Al"])

    def test_does_not_mutate_the_input(self):
        rows = [
            {"name": "Ana", "email": "ana@example.com", "plan": "pro"},
            {"name": "Ana", "email": "ana@example.com", "plan": "pro"},
        ]
        dedupe(rows)
        self.assertEqual(len(rows), 2)

    def test_empty(self):
        self.assertEqual(dedupe([]), [])


if __name__ == "__main__":
    unittest.main()
'''

# Tests the NEED. Never mentioned in the prompt, on purpose.
PROBES = '''
import unittest
from customers import dedupe

# One human, three spellings, exactly as they arrived from the two exports.
ONE_PERSON = [
    {"name": "Ana Diaz", "email": "Ana@Example.com", "plan": "pro"},
    {"name": "Ana Diaz", "email": "ana@example.com", "plan": "pro"},
    {"name": "Ana Diaz", "email": "a.na+crm@example.com", "plan": "pro"},
]


class TestIntent(unittest.TestCase):
    def test_case_variants_are_one_customer(self):
        self.assertEqual(len(dedupe(ONE_PERSON[:2])), 1)

    def test_gmail_style_aliases_are_one_customer(self):
        self.assertEqual(len(dedupe(ONE_PERSON)), 1)

    def test_different_humans_still_survive(self):
        rows = ONE_PERSON + [{"name": "Bo", "email": "bo@example.com", "plan": "free"}]
        self.assertEqual(len(dedupe(rows)), 2)


if __name__ == "__main__":
    unittest.main()
'''

# The boring version. Six lines, no model, no agent, written in the time it
# takes to read the ticket. It is here because a score with nothing to compare
# it against is unanchored, and because sometimes this is the right answer.
BASELINE = '''
def dedupe(rows):
    """Remove duplicate customers, preserving first-seen order."""
    seen = set()
    out = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
'''

TASK = Task(
    id="dedupe-customers",
    prompt=PROMPT,
    intent=INTENT,
    files={"customers.py": START},
    held_out_tests={"test_held_out.py": HELD_OUT},
    intent_probes={"test_probes.py": PROBES},
    allowed_files=["customers.py", "test_dedupe.py"],
    baseline={"customers.py": BASELINE},
    notes="Ask and need diverge. Correctness cannot detect that; probes can.",
)
