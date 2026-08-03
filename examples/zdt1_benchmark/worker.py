"""
worker.py

The external ZDT1 benchmark worker

Unlike the Iris example, this worker trains no model!
It evaluates the standard ZDT1 benchmark function, a
claaaaaaaasic multi-objective optimization problem
with a known Pareto front! Excellent!

The Iris worker says:

    "Can the optimizer successfully run a real training job?"

This worker answers:

    "Mmmmmm but is the search algorithm itself actually good at
    multi-objective search? Or does it just work."

Since we totally know the true Pareto front already, we can compare
algorithms against the real answer instead of just comparing two
plots by eye. It also runs almost instantly, making it practical
to compare thousands of trials across many random seeds HUZZAH!

Per project specs, his file intentionally lives outside
black_box_optimizer, which treats it exactly the same as any other
external worker program.

The original ZDT1 paper uses 30 variables. This worker deliberately uses
4 on purpose. As the number of variables grows, random search becomes
extremely unlikely to reach the true Pareto front, making it difficult
to meaningfully compare search algorithms, so for the sake of testing
I reduced! The problem stays multi-dimensional but still allows random
search to reach Pareto zone often enough that improvements are like
totally measurable!

"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

_NUM_VARIABLES = 4


def zdt1(variables: list[float]) -> tuple[float, float]:
    """Evaluate one candidate using the ZDT1 benchmark.

    ZDT1 is a standard multi-objective optimization problem with two
    objectives, both of which are minimized.

        x = (x1, ..., xn), each xi in [0, 1]

        f1(x) = x1
        g(x)  = 1 + 9 * sum(x2..xn) / (n - 1)
        f2(x) = g(x) * (1 - sqrt(f1(x) / g(x)))

    The true Pareto front is:

        f2 = 1 - sqrt(f1)

    It is reached only when every variable except x1 is exactly 0.
    """
    f1 = variables[0]
    g = 1.0 + 9.0 * sum(variables[1:]) / (_NUM_VARIABLES - 1)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return f1, f2


def write_metrics(metrics_path: Path, f1: float, f2: float) -> None:
    """Write one completed trial in the CSV format expected by metrics.py."""
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f1", "f2"])
        writer.writerow([f1, f2])


def main() -> None:
    """Read the candidate variables, evaluate ZDT1, and write the metrics."""
    parser = argparse.ArgumentParser()

    for index in range(1, _NUM_VARIABLES + 1):
        parser.add_argument(f"--x{index}", type=float, required=True)

    parser.add_argument("--metrics-out", type=str, required=True)
    args = parser.parse_args()

    variables = [
        getattr(args, f"x{index}")
        for index in range(1, _NUM_VARIABLES + 1)
    ]

    f1, f2 = zdt1(variables)
    write_metrics(Path(args.metrics_out), f1, f2)


if __name__ == "__main__":
    main()