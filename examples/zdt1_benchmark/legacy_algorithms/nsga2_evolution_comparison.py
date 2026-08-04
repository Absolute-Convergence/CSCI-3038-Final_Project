"""
nsga2_evolution_comparison.py

Compares all three steps of NSGA2's evolution (original -> +elitism ->
+elitism+polynomial mutation, i.e. the current shipped algorithm) against
random_search and against each other, on the ZDT1 benchmark.

This is a standalone script: it imports compare_search_algorithms.py's
building blocks (make_contract, make_worker_spec, hypervolume_2d,
mann_whitney_u, true_zdt1_hypervolume, _quartiles, write_raw_csv) but
never modifies that file, and defines its own algorithm-instance-driven
run function and its own multi-algorithm summary/chart helpers, since the
originals are hardcoded to the fixed two-name _ALGORITHMS tuple.

Two tests per NSGA2 version, each against random_search:
  Test 1: seeds shared between both algorithms, deterministic range.
          Reuses already-collected data from ~/Desktop/comparison_results*
          where available, only running what's missing.
  Test 2: seeds shared between both algorithms, drawn from a single fixed
          random set (not a deterministic range) -- generated once and
          reused identically across random_search and all three NSGA2
          versions.

Plus one more chart: all three NSGA2 versions plotted against each other
(no random_search), using Test 1's seed set, since that data already
covers all three versions once Test 1 finishes.
"""

from __future__ import annotations

import csv
import secrets
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

sys.path.insert(
    0, str(Path("/Users/emilytew/Desktop/Projects/CSCI-3038-Final_Project"))
)

from black_box_optimizer import runner
from black_box_optimizer.history import TrialHistory
from black_box_optimizer.pareto import build_pareto_front
from black_box_optimizer.records import build_trial_record
from black_box_optimizer.search.random_search import RandomSearch
from black_box_optimizer.search.nsga2 import NSGA2 as NSGA2Current
from examples.zdt1_benchmark import compare_search_algorithms as bench
from examples.zdt1_benchmark.legacy_algorithms.nsga2_original import (
    NSGA2Original,
)
from examples.zdt1_benchmark.legacy_algorithms.nsga2_elitism_only import (
    NSGA2ElitismOnly,
)

_DESKTOP = Path("/Users/emilytew/Desktop")
_TRIALS_PER_RUN = 500
_DETERMINISTIC_SEEDS = list(range(20))  # 0..19, matching the reused data
_OUTPUT_ROOT = (
    Path("/Users/emilytew/Desktop/Projects/CSCI-3038-Final_Project")
    / "examples"
    / "zdt1_benchmark"
    / "legacy_algorithms"
    / "evolution_comparison_results"
)

_VERSIONS = {
    "original": NSGA2Original,
    "elitism_only": NSGA2ElitismOnly,
    "elitism_polynomial_mutation": NSGA2Current,
}

_FUN_COLORS = {
    "random_search": "#FF6B6B",
    "original": "#FFD166",
    "elitism_only": "#06D6A0",
    "elitism_polynomial_mutation": "#7B2CBF",
}

_DESKTOP_SOURCES = {
    "original": _DESKTOP / "comparison_results",
    "elitism_only": _DESKTOP / "comparison_results_elitism",
    "elitism_polynomial_mutation": _DESKTOP / "comparison_results_polynomial_mutation",
}


def run_one_search_with_algorithm(algorithm, contract, worker_spec, trials):
    """Same shape as compare_search_algorithms.run_one_search(), but takes
    an already-constructed algorithm instance instead of a registry name,
    so it works with the variant classes that aren't registered anywhere.
    """
    history = TrialHistory()
    trace: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for trial_id in range(trials):
            proposal = algorithm.propose(contract, history.snapshot())
            if proposal.status != "candidate":
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
            trace.append(bench.hypervolume_2d(points, bench._REFERENCE_POINT))

    return trace


def _run_task(kind, name, seed, trials):
    """Top-level (picklable) unit of work for ProcessPoolExecutor."""
    contract = bench.make_contract()
    worker_spec = bench.make_worker_spec()
    if kind == "random_search":
        algorithm = RandomSearch(seed=seed)
    else:
        algorithm = _VERSIONS[name](seed=seed)
    return run_one_search_with_algorithm(algorithm, contract, worker_spec, trials)


def load_desktop_traces(name: str, source: Path) -> dict[tuple[str, int], list[float]]:
    """Load an already-collected raw_hypervolume_traces.csv from Desktop."""
    path = source / "raw_hypervolume_traces.csv"
    traces: dict[tuple[str, int], list[float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            algo = row["algorithm"]
            key_name = name if algo == "nsga2" else "random_search"
            key = (key_name, int(row["seed"]))
            traces.setdefault(key, [])
            traces[key].append(float(row["hypervolume"]))
    return traces


def run_batch(tasks, label):
    """Run a list of (kind, name, seed) tasks in parallel, return traces."""
    traces: dict[tuple[str, int], list[float]] = {}
    total = len(tasks)
    completed = 0
    started = time.perf_counter()
    print(f"\n--- {label}: {total} runs ---", flush=True)

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_run_task, kind, name, seed, _TRIALS_PER_RUN): (
                name,
                seed,
            )
            for kind, name, seed in tasks
        }
        for future in as_completed(futures):
            name, seed = futures[future]
            traces[(name, seed)] = future.result()
            completed += 1
            elapsed = time.perf_counter() - started
            print(
                f"[{completed}/{total}] {name} seed={seed} done "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )

    return traces


def print_summary_n(traces, true_hypervolume, names):
    print(f"\nTrue ZDT1 optimal hypervolume: {true_hypervolume:.4f}\n")
    print(f"{'algorithm':<32} {'seeds':<7} {'mean final HV':<15} "
          f"{'stdev':<10} {'% of optimal':<12}")
    for name in names:
        finals = [
            trace[-1]
            for (algo, _seed), trace in traces.items()
            if algo == name and trace
        ]
        if not finals:
            continue
        mean = statistics.fmean(finals)
        stdev = statistics.stdev(finals) if len(finals) > 1 else 0.0
        print(
            f"{name:<32} {len(finals):<7} {mean:<15.4f} {stdev:<10.4f} "
            f"{mean / true_hypervolume:<12.1%}"
        )


def print_pairwise_significance(traces, name_a, name_b):
    finals_a = [
        trace[-1] for (algo, _s), trace in traces.items() if algo == name_a and trace
    ]
    finals_b = [
        trace[-1] for (algo, _s), trace in traces.items() if algo == name_b and trace
    ]
    _u, p = bench.mann_whitney_u(finals_a, finals_b)
    verdict = "significant (p < 0.05)" if p < 0.05 else "NOT significant"
    print(f"{name_a} vs {name_b}: p = {p:.4f} -- {verdict}")


def plot_convergence_n(traces, true_hypervolume, output_path, names, title):
    figure = Figure(figsize=(8.5, 5.5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)

    for name in names:
        algorithm_traces = [
            trace for (algo, _s), trace in traces.items() if algo == name and trace
        ]
        if not algorithm_traces:
            continue
        shortest = min(len(t) for t in algorithm_traces)
        trimmed = [t[:shortest] for t in algorithm_traces]
        trial_numbers = list(range(1, shortest + 1))
        means = [
            statistics.fmean(t[i] for t in trimmed) for i in range(shortest)
        ]
        quartiles = [
            bench._quartiles([t[i] for t in trimmed]) for i in range(shortest)
        ]
        lower = [q1 for q1, _q3 in quartiles]
        upper = [q3 for _q1, q3 in quartiles]

        color = _FUN_COLORS.get(name)
        axes.plot(trial_numbers, means, label=name, color=color, linewidth=2)
        axes.fill_between(trial_numbers, lower, upper, color=color, alpha=0.18)

    axes.axhline(
        true_hypervolume, color="#333333", linestyle="--", linewidth=1,
        label="true optimal",
    )
    axes.set_xlabel("trials consumed")
    axes.set_ylabel("hypervolume")
    axes.set_title(title)
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path)
    print(f"Chart written to {output_path}")


def main() -> None:
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    contract = bench.make_contract()
    true_hv = bench.true_zdt1_hypervolume(bench._REFERENCE_POINT)

    # ---- Test 1: deterministic seeds 0-19, reusing Desktop data ----
    test1_traces: dict[tuple[str, int], list[float]] = {}

    # random_search doesn't change across versions -- reuse the original
    # dataset's random_search seeds 0-19 as the single Test 1 baseline.
    original_desktop = load_desktop_traces("original", _DESKTOP_SOURCES["original"])
    for key, trace in original_desktop.items():
        test1_traces[key] = trace  # includes both "original" and random_search

    missing_test1_tasks = []
    for name, source in _DESKTOP_SOURCES.items():
        if name == "original":
            continue  # already loaded above, has all 20 seeds
        desktop_data = load_desktop_traces(name, source)
        for key, trace in desktop_data.items():
            if key[0] == name:  # only take this version's nsga2 rows
                test1_traces[key] = trace
        have_seeds = {
            seed for (algo, seed) in desktop_data if algo == name
        }
        missing_seeds = set(_DETERMINISTIC_SEEDS) - have_seeds
        for seed in sorted(missing_seeds):
            missing_test1_tasks.append(("nsga2_variant", name, seed))

    if missing_test1_tasks:
        fresh = run_batch(missing_test1_tasks, "Test 1 top-up (missing seeds)")
        test1_traces.update(fresh)

    print("\n" + "=" * 70)
    print("TEST 1: deterministic seeds 0-19, shared between both algorithms")
    print("=" * 70)
    for name in _VERSIONS:
        pair_traces = {
            k: v
            for k, v in test1_traces.items()
            if k[0] in (name, "random_search")
        }
        print(f"\n-- {name} vs random_search --")
        print_summary_n(pair_traces, true_hv, [name, "random_search"])
        print_pairwise_significance(pair_traces, "random_search", name)
        plot_convergence_n(
            pair_traces, true_hv,
            _OUTPUT_ROOT / f"test1_{name}_vs_random_search.png",
            [name, "random_search"],
            f"ZDT1 Test 1 (seeds 0-19): {name} vs random_search",
        )

    write_test1_csv = _OUTPUT_ROOT / "test1_raw_traces.csv"
    with write_test1_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "seed", "trial_number", "hypervolume"])
        for (algo, seed), trace in sorted(test1_traces.items()):
            for trial_number, hv in enumerate(trace, start=1):
                writer.writerow([algo, seed, trial_number, hv])
    print(f"\nTest 1 raw traces written to {write_test1_csv}")

    # ---- 3-way NSGA2-only chart, reusing Test 1 data ----
    plot_convergence_n(
        test1_traces, true_hv,
        _OUTPUT_ROOT / "nsga2_versions_compared.png",
        list(_VERSIONS),
        "ZDT1: NSGA2 evolution (seeds 0-19) -- original vs elitism vs elitism+polynomial mutation",
    )
    print("\n-- NSGA2 versions vs each other (final hypervolume) --")
    print_summary_n(test1_traces, true_hv, list(_VERSIONS))
    version_names = list(_VERSIONS)
    for i in range(len(version_names)):
        for j in range(i + 1, len(version_names)):
            print_pairwise_significance(
                test1_traces, version_names[i], version_names[j]
            )

    # ---- Test 2: one shared set of 20 random (not sequential) seeds ----
    random_seeds = [secrets.randbelow(1_000_000) for _ in range(20)]
    print(f"\nTest 2 random seeds (shared across all 4): {random_seeds}")

    test2_tasks = [("random_search", "random_search", s) for s in random_seeds]
    for name in _VERSIONS:
        test2_tasks += [("nsga2_variant", name, s) for s in random_seeds]

    test2_traces = run_batch(test2_tasks, "Test 2 (shared random seeds)")

    print("\n" + "=" * 70)
    print("TEST 2: one shared set of 20 random seeds, all 4 algorithms")
    print("=" * 70)
    for name in _VERSIONS:
        pair_traces = {
            k: v
            for k, v in test2_traces.items()
            if k[0] in (name, "random_search")
        }
        print(f"\n-- {name} vs random_search --")
        print_summary_n(pair_traces, true_hv, [name, "random_search"])
        print_pairwise_significance(pair_traces, "random_search", name)
        plot_convergence_n(
            pair_traces, true_hv,
            _OUTPUT_ROOT / f"test2_{name}_vs_random_search.png",
            [name, "random_search"],
            f"ZDT1 Test 2 (shared random seeds): {name} vs random_search",
        )

    write_test2_csv = _OUTPUT_ROOT / "test2_raw_traces.csv"
    with write_test2_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "seed", "trial_number", "hypervolume"])
        for (algo, seed), trace in sorted(test2_traces.items()):
            for trial_number, hv in enumerate(trace, start=1):
                writer.writerow([algo, seed, trial_number, hv])
    print(f"\nTest 2 raw traces written to {write_test2_csv}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
