"""Three agents attempt one task. All three look fine in a demo.

Written as scripts rather than live model calls so the lesson reproduces
exactly, on any machine, with no API key and no cost. See scripted.py for
what that buys and what it does not.

Every one of these is a real solution someone has shipped. None is a straw man.
"""
from agenteng.scripted import ScriptedLLM, SelfCorrectingLLM, call, say

# --------------------------------------------------------------------------
# A. Ships fast. Dedupes on name, and tests it with a fixture where that works.
# --------------------------------------------------------------------------
A_HELPER = '''
def key_for(row):
    """Identity of a customer row."""
    return row["name"]
'''

A_IMPL = '''
from utils.helpers import key_for


def dedupe(rows):
    """Remove duplicate customers, preserving first-seen order."""
    seen = set()
    out = []
    for row in rows:
        k = key_for(row)
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out
'''

# Note what this fixture never contains: two different people with the
# same name. The agent chose the data that agrees with its implementation.
A_TESTS = '''
import unittest
from customers import dedupe


class TestDedupe(unittest.TestCase):
    def test_removes_duplicates(self):
        rows = [
            {"name": "Ana", "email": "ana@example.com", "plan": "pro"},
            {"name": "Ana", "email": "ana@example.com", "plan": "pro"},
            {"name": "Bo", "email": "bo@example.com", "plan": "free"},
        ]
        self.assertEqual(len(dedupe(rows)), 2)

    def test_order(self):
        rows = [
            {"name": "Zoe", "email": "zoe@example.com", "plan": "pro"},
            {"name": "Al", "email": "al@example.com", "plan": "free"},
        ]
        self.assertEqual([r["name"] for r in dedupe(rows)], ["Zoe", "Al"])

    def test_empty(self):
        self.assertEqual(dedupe([]), [])


if __name__ == "__main__":
    unittest.main()
'''

agent_a = ScriptedLLM(name="A-ships-fast", script=[
    call("read_file", path="customers.py"),
    call("write_file", path="utils/helpers.py", content=A_HELPER),
    call("write_file", path="customers.py", content=A_IMPL),
    call("write_file", path="test_dedupe.py", content=A_TESTS),
    call("run", cmd="python3 -m unittest test_dedupe -v"),
    say("Done. Extracted the identity logic into utils/helpers.py for reuse. All 3 tests pass."),
])

# --------------------------------------------------------------------------
# B. By the book. Dedupes the whole row. Correct, and beside the point.
# --------------------------------------------------------------------------
B_IMPL = '''
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

B_TESTS = '''
import unittest
from customers import dedupe


class TestDedupe(unittest.TestCase):
    def test_identical_rows_collapse(self):
        rows = [
            {"name": "Ana", "email": "ana@example.com", "plan": "pro"},
            {"name": "Ana", "email": "ana@example.com", "plan": "pro"},
        ]
        self.assertEqual(len(dedupe(rows)), 1)

    def test_same_name_different_person_survives(self):
        rows = [
            {"name": "Chris Lee", "email": "chris@acme.io", "plan": "pro"},
            {"name": "Chris Lee", "email": "clee@zenith.co", "plan": "free"},
        ]
        self.assertEqual(len(dedupe(rows)), 2)

    def test_input_not_mutated(self):
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

agent_b = ScriptedLLM(name="B-by-the-book", script=[
    call("read_file", path="customers.py"),
    call("write_file", path="customers.py", content=B_IMPL),
    call("write_file", path="test_dedupe.py", content=B_TESTS),
    call("run", cmd="python3 -m unittest test_dedupe -v"),
    say("Done. Dedupes on the full record, preserves order, does not mutate the input. 4 tests pass."),
])

# --------------------------------------------------------------------------
# C. Read the ticket. Dedupes on identity rather than on bytes.
# --------------------------------------------------------------------------
C_IMPL = '''
def _identity(row):
    """The human behind the row, not the row.

    Local-part dots and +tags are Gmail conventions. Applying them to every
    domain is a deliberate simplification for this exercise, and it is the
    kind of assumption that belongs in a decision record rather than in a
    silent one-liner.
    """
    email = (row.get("email") or "").strip().lower()
    local, _, domain = email.partition("@")
    local = local.split("+", 1)[0].replace(".", "")
    return (local, domain)


def dedupe(rows):
    """Remove duplicate customers by identity, preserving first-seen order."""
    seen = set()
    out = []
    for row in rows:
        key = _identity(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
'''

C_TESTS = '''
import unittest
from customers import dedupe


class TestDedupe(unittest.TestCase):
    def test_identical_rows_collapse(self):
        rows = [{"name": "Ana", "email": "ana@example.com", "plan": "pro"}] * 2
        self.assertEqual(len(dedupe(rows)), 1)

    def test_case_and_alias_variants_collapse(self):
        rows = [
            {"name": "Ana", "email": "Ana@Example.com", "plan": "pro"},
            {"name": "Ana", "email": "a.na+crm@example.com", "plan": "pro"},
        ]
        self.assertEqual(len(dedupe(rows)), 1)

    def test_distinct_people_survive(self):
        rows = [
            {"name": "Chris Lee", "email": "chris@acme.io", "plan": "pro"},
            {"name": "Chris Lee", "email": "clee@zenith.co", "plan": "free"},
        ]
        self.assertEqual(len(dedupe(rows)), 2)

    def test_order_and_no_mutation(self):
        rows = [
            {"name": "Zoe", "email": "zoe@example.com", "plan": "pro"},
            {"name": "Al", "email": "al@example.com", "plan": "free"},
            {"name": "Zoe", "email": "ZOE@example.com", "plan": "pro"},
        ]
        self.assertEqual([r["name"] for r in dedupe(rows)], ["Zoe", "Al"])
        self.assertEqual(len(rows), 3)

    def test_empty(self):
        self.assertEqual(dedupe([]), [])


if __name__ == "__main__":
    unittest.main()
'''

agent_c = ScriptedLLM(name="C-read-the-ticket", script=[
    call("read_file", path="customers.py"),
    call("run", cmd="ls"),
    call("write_file", path="customers.py", content=C_IMPL),
    call("write_file", path="test_dedupe.py", content=C_TESTS),
    call("run", cmd="python3 -m unittest test_dedupe -v"),
    say("Done. Deduped on normalised email identity, since the exports spell the same address several ways."),
])

ALL = [agent_a, agent_b, agent_c]


# --------------------------------------------------------------------------
# D. Writes a bug, runs its own check, fixes what the check finds.
#    The only agent here whose outcome depends on the harness.
# --------------------------------------------------------------------------
D_BUGGY = C_IMPL.replace('local = local.split("+", 1)[0].replace(".", "")',
                         'local = local.split("+", 1)[0]  # BUG: dots not stripped')

agent_d = SelfCorrectingLLM(
    name="D-self-correcting",
    draft=[
        call("read_file", path="customers.py"),
        call("write_file", path="customers.py", content=D_BUGGY),
        call("write_file", path="test_dedupe.py", content=C_TESTS),
    ],
    verify=call("run", cmd="python3 -m unittest test_dedupe 2>&1 | tail -3"),
    repair=[call("write_file", path="customers.py", content=C_IMPL)],
)

WITH_REPAIR = ALL + [agent_d]
