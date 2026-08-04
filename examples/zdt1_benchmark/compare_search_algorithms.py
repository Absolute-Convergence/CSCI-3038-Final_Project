"""
compare_search_algorithms.py

Runs RandomSearch and NSGA2 against Hyperloop's synthetic ZDT1 worker across
multiple seeds, and reports the hypervolume each
achieves relative to ZDT1's known, closed-form optimal Pareto front.

Nothing here is simulated: every trial launches synthetic_worker.py as a real
subprocess, exactly the same way runner.py always does. What's different
from the Iris example is that ZDT1 trials are near-instant (no model
training), so this can afford many seeds x many trials -- the two things
a single, slow, real-worker run can't give you:

  - Multiple independent seeds per algorithm, so "NSGA2 looks better" is
    a real trend across runs, not one lucky (or unlucky) draw.
  - A known ground truth. ZDT1's true optimal front is f2 = 1 - sqrt(f1),
    so "how good is this front" has an actual answer (hypervolume as a
    fraction of the true optimal hypervolume), not just two point clouds
    eyeballed side by side.

Usage:
    python -m examples.zdt1_benchmark.compare_search_algorithms
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from black_box_optimizer import runner
from black_box_optimizer.history import TrialHistory
from black_box_optimizer.models import (
    AlgorithmSpec,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    WorkerSpec,
)
from black_box_optimizer.pareto import build_pareto_front
from black_box_optimizer.records import build_trial_record
from black_box_optimizer.search.registry import create_algorithm
from hyperloop_workers import synthetic_worker

_WORKER_PATH = Path(synthetic_worker.__file__).resolve()
# Must match synthetic_worker.py's own _NUM_VARIABLES; see its docstring
# for why this is 4, not the original ZDT1 paper's 30.
_NUM_VARIABLES = 4

# Worse than any point ZDT1 can produce on either objective, so every real
# trial contributes some dominated area relative to it. Standard choice
# for ZDT1 hypervolume in the literature.
_REFERENCE_POINT = (1.1, 1.1)

_ALGORITHMS = ("random_search", "nsga2")
_DEFAULT_SEED_COUNT = 10
_DEFAULT_TRIALS_PER_RUN = 500

_OUTPUT_DIR = Path(__file__).parent / "comparison_results"


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the configurable search-efficacy benchmark interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare Random Search and NSGA-II against ZDT1 using real "
            "synthetic-worker subprocess trials."
        )
    )
    parser.add_argument(
        "--seeds",
        type=_positive_integer,
        default=_DEFAULT_SEED_COUNT,
        help="independent seeds per algorithm (default: 10)",
    )
    parser.add_argument(
        "--trials-per-run",
        type=_positive_integer,
        default=_DEFAULT_TRIALS_PER_RUN,
        help="worker trials per algorithm/seed run (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="directory for raw CSV and convergence chart",
    )
    return parser


def make_contract() -> OptimizationContract:
    """Build the bounded ZDT1 space with two minimized objectives."""
    parameters = tuple(
        ParameterDefinition(f"x{i}", ParameterKind.FLOAT, 0.0, 1.0)
        for i in range(1, _NUM_VARIABLES + 1)
    )
    objectives = (
        Objective("f1", Direction.MINIMIZE),
        Objective("f2", Direction.MINIMIZE),
    )
    return OptimizationContract(parameters=parameters, objectives=objectives)


def make_worker_spec() -> WorkerSpec:
    return WorkerSpec(
        command=(sys.executable, str(_WORKER_PATH)),
        metrics_argument="--metrics-out",
        # ZDT1 trials are near-instant (pure arithmetic, no training) --
        # generous only to absorb subprocess-launch variance, not real work.
        timeout_seconds=10.0,
    )


def hypervolume_2d(
    points: list[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    """Dominated hypervolume for 2D minimization, relative to `reference`.

    Standard sort-and-sweep algorithm. Sort candidate points by the first
    objective ascending; each point "owns" the x-range up to the next
    point that actually improves on the best second-objective value seen
    so far. Points at or worse than the reference on either objective
    contribute nothing. Works on any point set, not just an already-
    non-dominated front -- a dominated point simply never improves on
    best_y and is skipped.
    """
    ref_x, ref_y = reference
    candidates = sorted(
        (point for point in points if point[0] < ref_x and point[1] < ref_y),
        key=lambda point: point[0],
    )
    if not candidates:
        return 0.0

    area = 0.0
    previous_x: float | None = None
    best_y = math.inf
    for x, y in candidates:
        if y < best_y:
            if previous_x is not None:
                area += (x - previous_x) * (ref_y - best_y)
            previous_x = x
            best_y = y
    area += (ref_x - previous_x) * (ref_y - best_y)
    return area


def true_zdt1_hypervolume(
    reference: tuple[float, float],
    samples: int = 20_000,
) -> float:
    """Hypervolume of ZDT1's true optimal front (f2 = 1 - sqrt(f1)).

    Computed the same way as every other hypervolume in this file -- a
    dense, piecewise-linear sample of the known closed-form front fed
    through hypervolume_2d -- rather than a separately hand-derived
    formula, so there's only one hypervolume implementation to trust.
    """
    points = [
        (f1, 1.0 - math.sqrt(f1))
        for f1 in (i / samples for i in range(samples + 1))
    ]
    return hypervolume_2d(points, reference)


def run_one_search(
    algorithm_name: str,
    seed: int,
    contract: OptimizationContract,
    worker_spec: WorkerSpec,
    trials: int,
) -> list[float]:
    """Run one full seeded search, returning hypervolume after each trial."""
    algorithm = create_algorithm(AlgorithmSpec(name=algorithm_name, seed=seed))
    history = TrialHistory()
    trace: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for trial_id in range(trials):
            proposal = algorithm.propose(contract, history.snapshot())
            if proposal.status != "candidate":
                # Shouldn't happen on a continuous 4-dimensional float
                # space, but stop cleanly rather than crash if it ever does.
                break

            metrics_path = tmp_path / f"trial_{trial_id}.csv"
            execution_result = runner.execute(
                worker_spec, proposal.candidate, metrics_path
            )
            record = build_trial_record(
                proposal.candidate, trial_id, metrics_path, execution_result
            )
            history.append(record)

            front = build_pareto_front(history.snapshot(), contract)
            points = [
                (record.metrics["f1"], record.metrics["f2"])
                for record in front.records
            ]
            trace.append(hypervolume_2d(points, _REFERENCE_POINT))

    return trace


def write_raw_csv(
    path: Path, traces: dict[tuple[str, int], list[float]]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "seed", "trial_number", "hypervolume"])
        for (algorithm_name, seed), trace in traces.items():
            for trial_number, hypervolume in enumerate(trace, start=1):
                writer.writerow(
                    [algorithm_name, seed, trial_number, hypervolume]
                )


def print_summary(
    traces: dict[tuple[str, int], list[float]], true_hypervolume: float
) -> None:
    print(f"\nTrue ZDT1 optimal hypervolume (reference {_REFERENCE_POINT}): "
          f"{true_hypervolume:.4f}\n")
    print(f"{'algorithm':<15} {'seeds':<7} {'mean final HV':<15} "
          f"{'stdev':<10} {'% of optimal':<12}")
    for algorithm_name in _ALGORITHMS:
        finals = [
            trace[-1]
            for (name, _seed), trace in traces.items()
            if name == algorithm_name and trace
        ]
        mean = statistics.fmean(finals)
        stdev = statistics.stdev(finals) if len(finals) > 1 else 0.0
        pct = 100.0 * mean / true_hypervolume
        print(f"{algorithm_name:<15} {len(finals):<7} {mean:<15.4f} "
              f"{stdev:<10.4f} {pct:<12.1f}")


def plot_convergence(
    traces: dict[tuple[str, int], list[float]],
    true_hypervolume: float,
    output_path: Path,
) -> None:
    """Mean hypervolume vs. trial count per algorithm, averaged over seeds,
    with a min-max band across seeds and a dashed true-optimal reference
    line."""
    figure = Figure(figsize=(8.0, 5.5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)

    colors = {"random_search": "#d95f02", "nsga2": "#1b9e77"}
    for algorithm_name in _ALGORITHMS:
        algorithm_traces = [
            trace
            for (name, _seed), trace in traces.items()
            if name == algorithm_name and trace
        ]
        shortest = min(len(trace) for trace in algorithm_traces)
        trimmed = [trace[:shortest] for trace in algorithm_traces]

        trial_numbers = list(range(1, shortest + 1))
        means = [
            statistics.fmean(trace[i] for trace in trimmed)
            for i in range(shortest)
        ]
        minimums = [
            min(trace[i] for trace in trimmed) for i in range(shortest)
        ]
        maximums = [
            max(trace[i] for trace in trimmed) for i in range(shortest)
        ]

        color = colors.get(algorithm_name, None)
        axes.plot(trial_numbers, means, label=algorithm_name, color=color)
        axes.fill_between(
            trial_numbers, minimums, maximums, color=color, alpha=0.15
        )

    axes.axhline(
        true_hypervolume,
        color="black",
        linestyle="--",
        linewidth=1,
        label="true optimal",
    )
    axes.set_xlabel("trials consumed")
    axes.set_ylabel("hypervolume")
    axes.set_title("ZDT1: NSGA2 vs. random_search")
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path)


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    seeds = range(arguments.seeds)
    trials_per_run = arguments.trials_per_run
    output_directory = arguments.output_dir.resolve()
    contract = make_contract()
    worker_spec = make_worker_spec()
    true_hypervolume = true_zdt1_hypervolume(_REFERENCE_POINT)

    output_directory.mkdir(parents=True, exist_ok=True)

    traces: dict[tuple[str, int], list[float]] = {}
    total_runs = len(_ALGORITHMS) * len(seeds)
    total_worker_trials = total_runs * trials_per_run
    completed = 0
    started = time.perf_counter()
    print(
        f"Starting {total_worker_trials:,} real worker trials: "
        f"{len(_ALGORITHMS)} algorithms x {len(seeds)} seeds x "
        f"{trials_per_run} trials.",
        flush=True,
    )

    for algorithm_name in _ALGORITHMS:
        for seed in seeds:
            traces[(algorithm_name, seed)] = run_one_search(
                algorithm_name,
                seed,
                contract,
                worker_spec,
                trials_per_run,
            )
            completed += 1
            elapsed = time.perf_counter() - started
            print(
                f"[{completed}/{total_runs}] {algorithm_name} seed={seed} "
                f"done ({elapsed:.1f}s elapsed)",
                flush=True,
            )

    write_raw_csv(output_directory / "raw_hypervolume_traces.csv", traces)
    print_summary(traces, true_hypervolume)
    plot_convergence(
        traces,
        true_hypervolume,
        output_directory / "hypervolume_comparison.png",
    )
    print(f"\nRaw data and chart written to {output_directory}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
