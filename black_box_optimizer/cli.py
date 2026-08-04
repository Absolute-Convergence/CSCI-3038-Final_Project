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


# EMILY ADDITION FOR BEAUTIFICATION PURPOSES
#
# A real run against a real worker can take minutes with no output at all,
# which looks indistinguishable from a hang. This renders a single
# self-overwriting progress line as trials complete, and remembers the best
# value seen per objective so the final summary can report it.
class _ProgressReporter:
    _BAR_WIDTH = 24

    def __init__(self) -> None:
        self._max_trials: int | None = None
        self._contract: OptimizationContract | None = None
        self._best: dict[str, tuple[float, int]] = {}
        self._failure_counts: dict[str, int] = {}
        self._failure_examples: dict[str, str] = {}
        self._started_at = monotonic()
        self._printed = False

    def set_target(
        self, max_trials: int, contract: OptimizationContract
    ) -> None:
        self._max_trials = max_trials
        self._contract = contract

    def on_trial_complete(self, record: TrialRecord) -> None:
        eligible = self._contract is not None and is_eligible(
            record, self._contract
        )
        if eligible:
            for objective in self._contract.objectives:
                value = record.metrics[objective.metric_name]
                current = self._best.get(objective.metric_name)
                is_better = current is None or (
                    value > current[0]
                    if objective.direction == Direction.MAXIMIZE
                    else value < current[0]
                )
                if is_better:
                    self._best[objective.metric_name] = (
                        value,
                        record.trial_id,
                    )
        else:
            reason = self._failure_reason(record)
            self._failure_counts[reason] = (
                self._failure_counts.get(reason, 0) + 1
            )
            if reason not in self._failure_examples and record.error_message:
                self._failure_examples[reason] = record.error_message

        self._render(record.trial_id + 1)

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

    def print_best_summary(self) -> None:
        if self._contract is None:
            return
        for objective in self._contract.objectives:
            entry = self._best.get(objective.metric_name)
            if entry is None:
                continue
            value, trial_id = entry
            print(
                f"Best {objective.metric_name}: {value:.4f} "
                f"(trial {trial_id})"
            )

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
        print("Optimization cancelled during initialization.", file=sys.stderr)
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
    print(f"Status: {result.status}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Trials attempted: {result.attempted_count}")
    print(f"Pareto trials: {result.pareto_count}")
    progress.print_failure_summary()
    progress.print_best_summary()

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
