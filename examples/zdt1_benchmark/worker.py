"""
worker.py

The external, synthetic ZDT1 benchmark worker.

ZDT1 (Zitzler, Deb, Thiele, 2000) is a standard, textbook multi-objective
optimization test problem. Unlike the Iris worker in examples/iris_torch,
this trains nothing -- it is two closed-form formulas over a vector of
real numbers.

This exists to answer a different question than the Iris example does.
Iris shows the pipeline processing a real training job end to end. ZDT1
answers "is the search algorithm itself actually good at multi-objective
search" -- ZDT1's true optimal Pareto front is known exactly in closed
form, so results can be measured against ground truth instead of just
eyeballing two fronts side by side. It is also fast enough (no model
training at all) to run thousands of trials across many random seeds in
minutes, which a real training worker can't offer.

IMPORTANT NOTE: same rule as examples/iris_torch/worker.py -- this file
lives outside black_box_optimizer on purpose. The optimizer never imports
it and does not know or care that it isn't training anything real.

The problem, exactly as defined in the original paper:

    x = (x1, ..., xn), each xi in [0, 1]

    f1(x) = x1
    g(x)  = 1 + 9 * sum(x2..xn) / (n - 1)
    f2(x) = g(x) * (1 - sqrt(f1(x) / g(x)))

Both objectives are minimized. The true optimal front is f2 = 1 - sqrt(f1)
for f1 in [0, 1], reached whenever every parameter except x1 is exactly 0
-- that is the only way to get g(x) = 1.

The original paper uses n = 30. This file deliberately uses fewer, found
by direct measurement rather than guessing: with 29 "nuisance" parameters
that all need to land near 0 simultaneously, a sampled mean of 29
uniform[0, 1] draws concentrates tightly around 0.5 (an ordinary
consequence of averaging many independent values), so g stays close to
5.5 almost every draw -- across 20,000 random samples in testing, the
single best one still only reached g=3.55, nowhere near 1. Measuring the
actual probability of a uniform random sample landing under the (1.1,
1.1) reference point (see compare_search_algorithms.py) at a few sizes:

    n=2  (1 nuisance var):   ~12.1%  chance per sample
    n=4  (3 nuisance vars):  ~1.16%  chance per sample
    n=6  (5 nuisance vars):  ~0.19%  chance per sample -- under 1 expected
                                      hit per 200 random trials
    n=10 (9 nuisance vars):  ~0.006% chance per sample -- effectively never

This project's mutation operator also resets a parameter to a fresh
uniform draw rather than nudging it towards a better value, which makes
gradual convergence on a high-dimensional nuisance space even harder than
it would be for a textbook NSGA-II with polynomial mutation. n = 4 (3
nuisance parameters) is small enough that random sampling still finds the
optimal region often enough for a real comparison, while staying a
genuine multi-dimensional problem rather than a trivial one.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

_NUM_VARIABLES = 4


def zdt1(variables: list[float]) -> tuple[float, float]:
    """Compute the two ZDT1 objectives for one candidate."""
    f1 = variables[0]
    g = 1.0 + 9.0 * sum(variables[1:]) / (_NUM_VARIABLES - 1)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return f1, f2


def write_metrics(metrics_path: Path, f1: float, f2: float) -> None:
    """Write one completed trial in the CSV format expected by metrics.py"""
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f1", "f2"])
        writer.writerow([f1, f2])


def main() -> None:
    """Read x1..x30, compute ZDT1, and write the resulting metrics."""
    parser = argparse.ArgumentParser()
    for index in range(1, _NUM_VARIABLES + 1):
        parser.add_argument(f"--x{index}", type=float, required=True)
    parser.add_argument("--metrics-out", type=str, required=True)
    args = parser.parse_args()

    variables = [
        getattr(args, f"x{index}") for index in range(1, _NUM_VARIABLES + 1)
    ]
    f1, f2 = zdt1(variables)

    write_metrics(Path(args.metrics_out), f1, f2)


if __name__ == "__main__":
    main()
