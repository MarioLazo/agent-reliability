# Agent Engineering: the course harness.
# Stdlib only. No virtualenv, no install step, no network.

.PHONY: verify quick test bench notebooks nb-check clean

## verify: the whole gate. Run this before every commit.
verify: test nb-check bench
	@echo "OK: harness tested, notebooks current, benchmark reproduces."

## quick: the fast loop. Tests only, no notebooks, no benchmark.
quick: test

## test: the harness grades code, so the harness gets graded too.
test:
	@python3 -m unittest discover -s tests -t . -q

## bench: run every agent against every task and print the table.
bench:
	@python3 -m bench.run

## notebooks: rebuild .ipynb from notebooks/src/*.py
notebooks:
	@python3 tools/nbbuild.py

## nb-check: fail if a committed .ipynb is stale, and run every notebook.
nb-check:
	@python3 tools/nbbuild.py --check
	@for f in notebooks/src/*.py; do \
		python3 "$$f" > /dev/null || { echo "FAILED: $$f"; exit 1; }; \
	done
	@echo "notebooks execute clean"

clean:
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
