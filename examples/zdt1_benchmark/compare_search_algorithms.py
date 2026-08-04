"""
compare_search_algorithms.py

Runs both search algos against Hyperloops synthetic ZDT1 worker across
many multiple seeds magoo then reports the hypervolume each one reaches!

Every trial still launches synthetic_worker.py through runner.py as a real
subprocess because the very arithmetic ZDT1 is fast enough that we can
afford a bunch of seeds and trials without waiting like for model training

That gives us two things one normal worker run can't:

  - Multiple seeds so one lucky run cant carry the whole conclusion
  - A known answer because ZDT1s optimal front is f2 = 1 - sqrt(f1)

Usage:
    python -m examples.zdt1_benchmark.compare_search_algorithms
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Must match synthetic_worker.py!
# This version uses four variables instead of the original paper's thirty
# so the benchmark stays lightweight
_NUM_VARIABLES = 4

# Slightly worse than any real ZDT1 point so every valid trial can
# contribute some area toward the hypervolume
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
    """Build the command line options for the benchmark"""
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
    parser.add_argument(
        "--jobs",
        type=_positive_integer,
        default=os.cpu_count() or 1,
        help=(
            "algorithm/seed runs to execute concurrently, each in its own "
            "process (default: os.cpu_count()); runs are fully independent, "
            "so this only affects wall-clock time, not results"
        ),
    )
    return parser


def make_contract() -> OptimizationContract:
    """Build the bounded ZDT1 search space and its two objectives."""
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
        # ZDT1 only does a little arithmetic so this timeout mostly gives
        # subprocess startup plenty of breathing room
        timeout_seconds=10.0,
    )


def hypervolume_2d(
    points: list[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    """Calculate dominated hypervolume for a 2D minimization problem

    Points are sorted by the first objective, then swept left to right.
    A point only adds area when it improves the best second objective
    value we've seen so far.

    Points outside the reference boundary add nothing, and dominated
    points naturally get skipped so the input doesn't need to already
    be a clean Pareto front.
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


def _standard_normal_cdf(z: float) -> float:
    """Return P Z less than or equal to z for a standard normal variable"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> tuple[float, float]:
    """Run a two-sided Mann Whitney U test (without scipy)

    This checks whether one algorithm generally produces higher results
    than the other, without assuming the samples are normally distributed.

    Returns the U statistic for sample_a and the two-sided p value

    The p value uses the normal approximation which works for the seed
    counts this benchmark is meant to run...tee tiny samples would need an
    exact permutation test instead!
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    labeled = sorted(
        [(value, "a") for value in sample_a]
        + [(value, "b") for value in sample_b]
    )

    # Give every value a rank from 1 through n
    # Tied values split the ranks they occupy
    ranks: list[float] = [0.0] * len(labeled)
    tie_group_sizes: list[int] = []
    index = 0
    while index < len(labeled):
        end = index
        while end < len(labeled) and labeled[end][0] == labeled[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[position] = average_rank
        tie_group_sizes.append(end - index)
        index = end

    rank_sum_a = sum(
        rank
        for rank, (_value, label) in zip(ranks, labeled)
        if label == "a"
    )
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0

    n_total = n_a + n_b
    mean_u = n_a * n_b / 2.0
    tie_correction = sum(size**3 - size for size in tie_group_sizes)
    if n_total <= 1:
        return u_a, 1.0
    variance_u = (n_a * n_b / 12.0) * (
        (n_total + 1) - tie_correction / (n_total * (n_total - 1))
    )
    if variance_u <= 0:
        # Every value is identical so there's no difference to detect
        return u_a, 1.0

    z_score = (u_a - mean_u) / math.sqrt(variance_u)
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z_score)))
    return u_a, min(1.0, p_value)


def true_zdt1_hypervolume(
    reference: tuple[float, float],
    samples: int = 20_000,
) -> float:
    """Approximate the hypervolume of ZDT1s known optimal front.

    Sample the known front densely then feed it through the same
    hypervolume function everything else uses, so there's only one
    implementation to trust.
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
    """Run one seeded search and record hypervolume after every trial"""
    algorithm = create_algorithm(AlgorithmSpec(name=algorithm_name, seed=seed))
    history = TrialHistory()
    trace: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for trial_id in range(trials):
            proposal = algorithm.propose(contract, history.snapshot())
            if proposal.status != "candidate":
                # A continuous float space shouldn't run out but it's better to
                # stop cleanly than get funky if something changes
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
    """Write every algorithm seed trial and hypervolume result to CSV"""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "seed", "trial_number", "hypervolume"])
        for (algorithm_name, seed), trace in sorted(traces.items()):
            for trial_number, hypervolume in enumerate(trace, start=1):
                writer.writerow(
                    [algorithm_name, seed, trial_number, hypervolume]
                )


def print_summary(
    traces: dict[tuple[str, int], list[float]], true_hypervolume: float
) -> None:
    """Print the final results and significance test"""
    print(f"\nTrue ZDT1 optimal hypervolume (reference {_REFERENCE_POINT}): "
          f"{true_hypervolume:.4f}\n")
    print(f"{'algorithm':<15} {'seeds':<7} {'mean final HV':<15} "
          f"{'stdev':<10} {'% of optimal':<12}")
    finals_by_algorithm: dict[str, list[float]] = {}
    for algorithm_name in _ALGORITHMS:
        finals = [
            trace[-1]
            for (name, _seed), trace in traces.items()
            if name == algorithm_name and trace
        ]
        finals_by_algorithm[algorithm_name] = finals
        mean = statistics.fmean(finals)
        stdev = statistics.stdev(finals) if len(finals) > 1 else 0.0
        pct = 100.0 * mean / true_hypervolume
        print(f"{algorithm_name:<15} {len(finals):<7} {mean:<15.4f} "
              f"{stdev:<10.4f} {pct:<12.1f}")

    # Means can look different just from seed luck
    # This checks whether the result distributions are actually separated
    if len(_ALGORITHMS) == 2:
        first_name, second_name = _ALGORITHMS
        _u_statistic, p_value = mann_whitney_u(
            finals_by_algorithm[first_name], finals_by_algorithm[second_name]
        )
        verdict = (
            "statistically significant (p < 0.05)"
            if p_value < 0.05
            else "NOT statistically significant at the 0.05 level"
        )
        print(
            f"\nMann-Whitney U test on final hypervolume, "
            f"{first_name} vs. {second_name}: p = {p_value:.4f} -- {verdict}"
        )


# These are trial counts a real worker might actually be able to afford
# A training job can take minutes per trial while this one takes almost
# no time so the full benchmark budget wouldn't be realistic there
_REALISTIC_BUDGET_CHECKPOINTS = (10, 25, 50, 100)


def print_budget_checkpoints(
    traces: dict[tuple[str, int], list[float]],
) -> None:
    """Show whether NSGA2's advantage appears at smaller trial budgets.

    Huge trial budgets are only practical because ZDT1 is synthetic.
    The useful question is whether the advantage shows up before a real
    slow worker gets outrageously expensive

    This reuses the traces we already collected so it doesn't launch any
    extra worker trials.
    """
    if len(_ALGORITHMS) != 2:
        return

    first_name, second_name = _ALGORITHMS
    max_affordable_trials = min(
        (len(trace) for trace in traces.values() if trace), default=0
    )

    print("\nDoes the advantage hold at a realistic (small) trial budget?")
    print(
        f"{'trials':<8} {first_name:<16} {second_name:<10} {'p-value':<10}"
    )
    for checkpoint in _REALISTIC_BUDGET_CHECKPOINTS:
        if checkpoint > max_affordable_trials:
            break

        values_by_algorithm = {
            algorithm_name: [
                trace[checkpoint - 1]
                for (name, _seed), trace in traces.items()
                if name == algorithm_name and len(trace) >= checkpoint
            ]
            for algorithm_name in _ALGORITHMS
        }
        first_mean = statistics.fmean(values_by_algorithm[first_name])
        second_mean = statistics.fmean(values_by_algorithm[second_name])
        _u_statistic, p_value = mann_whitney_u(
            values_by_algorithm[first_name], values_by_algorithm[second_name]
        )
        print(
            f"{checkpoint:<8} {first_mean:<16.4f} {second_mean:<10.4f} "
            f"{p_value:<10.4f}"
        )


def _quartiles(values: Sequence[float]) -> tuple[float, float]:
    """Return Q1 and Q3 for the values.

    One value doesn't have any spread so it becomes v and v instead of
    making statistics.quantiles complain.
    """
    if len(values) < 2:
        (only,) = values
        return only, only
    first_quartile, _median, third_quartile = statistics.quantiles(
        values, n=4
    )
    return first_quartile, third_quartile


def plot_convergence(
    traces: dict[tuple[str, int], list[float]],
    true_hypervolume: float,
    output_path: Path,
) -> None:
    """Plot mean hypervolume across trials for each algorithm.

    The shaded band shows the interquartile range across seeds instead
    of min and max, so one very lucky or cursed seed can't take over the
    whole chart.
    """
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
        quartiles = [
            _quartiles([trace[i] for trace in trimmed])
            for i in range(shortest)
        ]
        lower_quartiles = [q1 for q1, _q3 in quartiles]
        upper_quartiles = [q3 for _q1, q3 in quartiles]

        color = colors.get(algorithm_name, None)
        axes.plot(trial_numbers, means, label=algorithm_name, color=color)
        axes.fill_between(
            trial_numbers,
            lower_quartiles,
            upper_quartiles,
            color=color,
            alpha=0.15,
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
    axes.set_title(
        "ZDT1: NSGA2 vs. random_search"
    )
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
    tasks = [
        (algorithm_name, seed)
        for algorithm_name in _ALGORITHMS
        for seed in seeds
    ]
    total_runs = len(tasks)
    total_worker_trials = total_runs * trials_per_run
    completed = 0
    started = time.perf_counter()
    print(
        f"Starting {total_worker_trials:,} real worker trials: "
        f"{len(_ALGORITHMS)} algorithms x {len(seeds)} seeds x "
        f"{trials_per_run} trials ({arguments.jobs} run(s) at a time).",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=arguments.jobs) as executor:
        futures = {
            executor.submit(
                run_one_search,
                algorithm_name,
                seed,
                contract,
                worker_spec,
                trials_per_run,
            ): (algorithm_name, seed)
            for algorithm_name, seed in tasks
        }
        for future in as_completed(futures):
            algorithm_name, seed = futures[future]
            traces[(algorithm_name, seed)] = future.result()
            completed += 1
            elapsed = time.perf_counter() - started
            print(
                f"[{completed}/{total_runs}] {algorithm_name} seed={seed} "
                f"done ({elapsed:.1f}s elapsed)",
                flush=True,
            )

    write_raw_csv(output_directory / "raw_hypervolume_traces.csv", traces)
    print_summary(traces, true_hypervolume)
    print_budget_checkpoints(traces)
    plot_convergence(
        traces,
        true_hypervolume,
        output_directory / "hypervolume_comparison.png",
    )
    print(f"\nRaw data and chart written to {output_directory}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
