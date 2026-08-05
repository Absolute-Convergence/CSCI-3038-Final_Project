"""Command-line composition for a local Black Box Optimizer run."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from time import monotonic

from black_box_optimizer.application import initialize_application
from black_box_optimizer.config_loader import ConfigurationError
from black_box_optimizer.models import Direction, OptimizationContract
from black_box_optimizer.pareto import is_eligible
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.reporting import ReportingError
from black_box_optimizer.results import OptimizationResult


# EMILY ADDITION FOR BEAUTIFICATION PURPOSES
#
# A real run against a real worker can take minutes with no output at all,
# which looks indistinguishable from a hang. This renders a single
# self-overwriting progress line as trials complete.
class _ProgressReporter:
    _BAR_WIDTH = 24

    def __init__(self) -> None:
        self._max_trials: int | None = None
        self._contract: OptimizationContract | None = None
        self._failure_counts: dict[str, int] = {}
        self._failure_examples: dict[str, str] = {}
        self._started_at = monotonic()
        self._printed = False
        self.completed_count = 0

    def set_target(
        self, max_trials: int, contract: OptimizationContract
    ) -> None:
        self._max_trials = max_trials
        self._contract = contract

    def on_trial_complete(self, record: TrialRecord) -> None:
        eligible = self._contract is not None and is_eligible(
            record, self._contract
        )
        if not eligible:
            reason = self._failure_reason(record)
            self._failure_counts[reason] = (
                self._failure_counts.get(reason, 0) + 1
            )
            if reason not in self._failure_examples and record.error_message:
                self._failure_examples[reason] = record.error_message

        self.completed_count = record.trial_id + 1
        self._render(self.completed_count)

    @staticmethod
    def _failure_reason(record: TrialRecord) -> str:
        if record.execution_status != "completed":
            return record.execution_status
        if record.metrics_status != "valid":
            return f"metrics_{record.metrics_status}"
        # execution succeeded and the metrics file parsed fine, so the
        # only remaining way is_eligible() can fail is a declared
        # objective's metric key missing from this record.
        return "missing_objective_metric"

    def _render(self, completed: int) -> None:
        self._printed = True
        total = self._max_trials or completed
        fraction = min(1.0, completed / total)
        filled = int(fraction * self._BAR_WIDTH)
        bar = ("=" * filled).ljust(self._BAR_WIDTH)
        elapsed = int(monotonic() - self._started_at)
        minutes, seconds = divmod(elapsed, 60)
        elapsed_text = f"{minutes}m{seconds:02d}s"
        failed = sum(self._failure_counts.values())
        failed_text = f" ({failed} failed)" if failed else ""
        sys.stdout.write(
            f"\rTrial {completed}/{total} [{bar}] elapsed {elapsed_text}"
            f"{failed_text}"
        )
        sys.stdout.flush()

    def finish_line(self) -> None:
        if self._printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def print_failure_summary(self) -> None:
        total_failed = sum(self._failure_counts.values())
        if total_failed == 0:
            return
        print(f"\n{total_failed} trial(s) did not produce usable metrics:")
        for reason, count in sorted(self._failure_counts.items()):
            example = self._failure_examples.get(reason)
            detail = f" -- e.g. {example}" if example else ""
            print(f"  {reason}: {count}{detail}")
        print("See history.csv in the run directory for full details.")


# EMILY ADDITION FOR BEAUTIFICATION PURPOSES
#
# Deliberately not tracked incrementally during the run the way progress
# and failures are. A per-trial "best so far" tracker can only compare
# one objective at a time, so it can end up reporting a trial that a
# *later* trial actually dominates (identical-or-better on every other
# objective too) as if it were undominated -- the run's real Pareto
# front is the only thing that can answer "is this genuinely one of the
# best trade-offs," so this is computed once, after the run, straight
# from result.pareto_front -- the same authoritative source
# reporting.py's chart already draws its own best-point labels from.
def _print_best_summary(
    result: OptimizationResult, contract: OptimizationContract
) -> None:
    pareto_records = result.pareto_front.records
    if not pareto_records:
        return
    for objective in contract.objectives:
        best = max if objective.direction == Direction.MAXIMIZE else min
        best_record = best(
            pareto_records,
            key=lambda record: record.metrics[objective.metric_name],
        )
        value = best_record.metrics[objective.metric_name]
        print(
            f"Best {objective.metric_name}: {value:.4f} "
            f"(trial {best_record.trial_id})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, prog="python -m black_box_optimizer")


def installed_main(argv: Sequence[str] | None = None) -> int:
    """Run through the installed ``hyperloop-optimizer`` command."""
    return _run(argv, prog="hyperloop-optimizer")


def _run(argv: Sequence[str] | None, *, prog: str) -> int:
    parser = _build_parser(prog)
    arguments = parser.parse_args(argv)

    # EMILY ADDITION FOR BEAUTIFICATION PURPOSES
    progress = _ProgressReporter()

    try:
        session = initialize_application(
            arguments.configuration,
            arguments.output_dir,
            on_trial_complete=progress.on_trial_complete,
        )
        progress.set_target(
            session.configuration.stop_policy.max_trials,
            session.configuration.optimization,
        )
        result = session.run()
    except KeyboardInterrupt:
        progress.finish_line()
        if progress.completed_count:
            print(
                f"Optimization cancelled while a trial was in progress. "
                f"{progress.completed_count} trial(s) completed and were "
                "safely recorded before the interrupt, but this run's "
                "final report was not written.",
                file=sys.stderr,
            )
        else:
            print(
                "Optimization cancelled during initialization.",
                file=sys.stderr,
            )
        return 130
    except ConfigurationError as error:
        print(error, file=sys.stderr)
        return 2
    except (OSError, ReportingError) as error:
        progress.finish_line()
        print(f"Optimization failed: {error}", file=sys.stderr)
        return 1

    progress.finish_line()
    print(f"Run directory: {session.run_directory.path}")
    print(
        f"Seed: {session.configuration.algorithm.seed} "
        "(supply this in algorithm.seed to reproduce this exact run)"
    )
    print(f"Status: {result.status}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Trials attempted: {result.attempted_count}")
    print(f"Pareto trials: {result.pareto_count}")
    progress.print_failure_summary()
    _print_best_summary(result, session.configuration.optimization)

    if result.status == "cancelled":
        return 130
    if result.status == "failed":
        return 1
    return 0


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Optimize one opaque local worker from a validated JSON project "
            "configuration."
        ),
    )
    parser.add_argument(
        "configuration",
        help="path to the project configuration JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="base directory for a new unique run (default: runs)",
    )
    return parser
