"""Command-line composition for a local Black Box Optimizer run."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from black_box_optimizer.application import initialize_application
from black_box_optimizer.config_loader import ConfigurationError
from black_box_optimizer.reporting import ReportingError


def main(argv: Sequence[str] | None = None) -> int:
    return _run(argv, prog="python -m black_box_optimizer")


def installed_main(argv: Sequence[str] | None = None) -> int:
    """Run through the installed ``hyperloop-optimizer`` command."""
    return _run(argv, prog="hyperloop-optimizer")


def _run(argv: Sequence[str] | None, *, prog: str) -> int:
    parser = _build_parser(prog)
    arguments = parser.parse_args(argv)

    try:
        session = initialize_application(
            arguments.configuration,
            arguments.output_dir,
        )
        result = session.run()
    except KeyboardInterrupt:
        print("Optimization cancelled during initialization.", file=sys.stderr)
        return 130
    except ConfigurationError as error:
        print(error, file=sys.stderr)
        return 2
    except (OSError, ReportingError) as error:
        print(f"Optimization failed: {error}", file=sys.stderr)
        return 1

    print(f"Run directory: {session.run_directory.path}")
    print(f"Status: {result.status}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Trials attempted: {result.attempted_count}")
    print(f"Pareto trials: {result.pareto_count}")

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
