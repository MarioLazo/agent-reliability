"""Three agents. Each one does its job correctly. The system is wrong.

This is the half of the delegation lesson that a probability cannot teach.
The simulator tells you 0.9 cubed is 0.729. It does not tell you what a
cascade *looks like* from the inside, which is the thing that makes engineers
stop adding agents.

The setup is a three-stage pipeline of the shape everybody builds:

    architect  ->  writes the interface
    developer  ->  implements against the interface
    tester     ->  writes tests from the interface

The ticket says: apply the discount to the price.
The architect writes a spec for a function that returns the discount AMOUNT.

Everything downstream is then correct. The developer implements the spec
faithfully. The tester tests the spec faithfully. Every test passes. Read the
transcript end to end and there is nothing in it that looks like a failure,
because at no point did any agent do anything wrong.

The error entered at stage one and every later stage propagated it with
perfect fidelity. That is what a cascade is, and it is why decomposition, not
implementation, is the decision that matters.
"""
from agenteng.scripted import ScriptedLLM, call, say

TICKET = """Customers get a percentage discount at checkout.
Add `apply_discount(price, discount)` so the checkout can show what the
customer pays."""

# --- stage 1: the architect ------------------------------------------------
# One word wrong in a docstring. Everything else follows from it.
SPEC = '''
"""Interface for the checkout discount feature.

apply_discount(price: float, discount: float) -> float
    Return the discount amount for `price`, where `discount` is a
    percentage between 0 and 100.

    Example: apply_discount(100.0, 20.0) -> 20.0
"""
'''

architect = ScriptedLLM(name="1-architect", script=[
    call("read_file", path="TICKET.md"),
    call("write_file", path="spec.py", content=SPEC),
    say("Interface defined. apply_discount(price, discount) returns a float, discount as a percentage."),
])

# --- stage 2: the developer ------------------------------------------------
IMPL = '''
def apply_discount(price, discount):
    """Return the discount amount for `price`.

    `discount` is a percentage between 0 and 100. See spec.py.
    """
    if not 0 <= discount <= 100:
        raise ValueError("discount must be a percentage between 0 and 100")
    return price * discount / 100
'''

developer = ScriptedLLM(name="2-developer", script=[
    call("read_file", path="spec.py"),
    call("write_file", path="checkout.py", content=IMPL),
    say("Implemented to spec, including the range check the spec implies."),
])

# --- stage 3: the tester ---------------------------------------------------
TESTS = '''
import unittest
from checkout import apply_discount


class TestApplyDiscount(unittest.TestCase):
    def test_matches_the_spec_example(self):
        self.assertEqual(apply_discount(100.0, 20.0), 20.0)

    def test_zero_discount(self):
        self.assertEqual(apply_discount(100.0, 0.0), 0.0)

    def test_full_discount(self):
        self.assertEqual(apply_discount(100.0, 100.0), 100.0)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            apply_discount(100.0, 150.0)


if __name__ == "__main__":
    unittest.main()
'''

tester = ScriptedLLM(name="3-tester", script=[
    call("read_file", path="spec.py"),
    call("write_file", path="test_checkout.py", content=TESTS),
    call("run", cmd="python3 -m unittest test_checkout -v"),
    say("4 tests, all passing. Coverage matches the spec including the error case."),
])

PIPELINE = [architect, developer, tester]

# Written by the person who filed the ticket, and shown to nobody in the chain.
HELD_OUT = '''
import unittest
from checkout import apply_discount


class TestWhatTheCustomerPays(unittest.TestCase):
    """The ticket said: show what the customer pays."""

    def test_customer_pays_the_discounted_price(self):
        self.assertEqual(apply_discount(100.0, 20.0), 80.0)

    def test_no_discount_means_full_price(self):
        self.assertEqual(apply_discount(100.0, 0.0), 100.0)


if __name__ == "__main__":
    unittest.main()
'''


def run_pipeline(workdir=None):
    """Run all three stages in one shared workspace and return the trajectories."""
    import pathlib
    import tempfile
    from agenteng.loop import run_agent

    wd = pathlib.Path(workdir or tempfile.mkdtemp(prefix="cascade-"))
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "TICKET.md").write_text(TICKET)

    trajectories = []
    for agent in PIPELINE:
        trajectories.append(run_agent(agent, workdir=wd))
    return wd, trajectories
