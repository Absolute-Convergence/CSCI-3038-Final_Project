"""Run-directory layout and atomic persistence of trial evidence."""

from __future__ import annotations

import csv
import os
import secrets
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from black_box_optimizer.models import OptimizationContract
from black_box_optimizer.pareto import is_eligible
from black_box_optimizer.records import TrialRecord

_TRIAL_DIRECTORY_DIGITS = 4


class CheckpointError(RuntimeError):
    """
    Raised when required run evidence cannot be atomically persisted.

    TDS section 10.3 treats this as fatal, but the in-memory TrialRecord
    appended before the checkpoint attempt is not lost. Callers should
    finalize with fatal_error instead of losing that evidence.
    """


# Fixed columns required by the TDS in order
# Any param and metric columns go in the middle
# error_message always comes last
_FIXED_COLUMNS = (
    "trial_id",
    "execution_status",
    "metrics_status",
    "execution_succeeded",
    "objective_eligible",
    "runtime_seconds",
    "exit_code",
    "timed_out",
)


class RunDirectory:
    """
    Owns one run's directory layout, per-trial paths, and history
    checkpoints.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._trials_path = path / "trials"
        self._trials_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The run's own directory."""
        return self._path

    def trial_directory(self, trial_id: int) -> Path:
        """Create, if needed, and return one trial's own directory."""
        directory = self._trials_path / self._trial_directory_name(trial_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise CheckpointError(
                f"Failed to create directory for trial_id {trial_id}: "
                f"{error}"
            ) from error
        return directory

    def metrics_path(self, trial_id: int) -> Path:
        """
        Return where one trial's metrics.csv belongs.

        The trial directory is created first if it does not already
        exist.
        """
        return self.trial_directory(trial_id) / "metrics.csv"

    def write_diagnostics(
        self,
        trial_id: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Persist complete decoded worker streams in the trial directory."""
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise TypeError("stdout and stderr must be strings")
        try:
            directory = self.trial_directory(trial_id)
            _atomic_write_text(directory / "stdout.txt", stdout)
            _atomic_write_text(directory / "stderr.txt", stderr)
        except CheckpointError:
            raise
        except Exception as error:
            raise CheckpointError(
                f"Failed to persist diagnostics for trial_id {trial_id}: "
                f"{error}"
            ) from error

    def checkpoint(
        self,
        history: Sequence[TrialRecord],
        contract: OptimizationContract,
    ) -> None:
        """
        Atomically replace history.csv with the newest flattened history.

        Everything is written to a temporary file first, then swapped
        into place with os.replace(). If something goes wrong halfway
        through, the previous checkpoint stays intact instead of being
        left half-written.

        If checkpointing fails, it is treated as fatal according to TDS
        section 10.3.
        """
        destination = self._path / "history.csv"
        fieldnames = _build_fieldnames(history, contract)

        trial_note = (
            f" (most recent trial_id: {history[-1].trial_id})"
            if history
            else ""
        )
        try:
            fd, temp_name = tempfile.mkstemp(
                dir=self._path, prefix="history.", suffix=".csv.tmp"
            )
        except OSError as error:
            raise CheckpointError(
                "Failed to create a temporary history checkpoint"
                f"{trial_note}: {error}"
            ) from error

        try:
            with os.fdopen(fd, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()

                for record in history:
                    writer.writerow(
                        _flatten_record(record, contract, fieldnames)
                    )

            os.replace(temp_name, destination)

        except Exception as error:
            _remove_temporary_file(temp_name)
            raise CheckpointError(
                "Failed to checkpoint history.csv; the previously "
                f"committed file (if any) was retained{trial_note}: "
                f"{error}"
            ) from error
        except BaseException:
            _remove_temporary_file(temp_name)
            raise

    @staticmethod
    def _trial_directory_name(trial_id: int) -> str:
        # KNOWN DEVIATION
        # The TDS example starts at trial_0001 but trial IDs are zero
        # indexed everywhere else in this project
        # Keeping that same numbering here avoids creating an off by one
        # just to match what looks like an illustrative example
        return f"trial_{trial_id:0{_TRIAL_DIRECTORY_DIGITS}d}"


def create_run_directory(base_directory: str | Path) -> RunDirectory:
    """
    Create a brand-new run directory.

    Runs are never resumed or overwritten here. Every optimization gets
    its own uniquely named directory so previous runs stay untouched.
    """
    base = Path(base_directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = secrets.token_hex(3)
    run_path = base / f"run_{timestamp}_{suffix}"

    run_path.mkdir(parents=True, exist_ok=False)

    return RunDirectory(run_path)


def _build_fieldnames(
    history: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> list[str]:
    """
    Build the history.csv column order.

    The fixed columns come first, followed by every declared parameter,
    every metric observed so far, and finally error_message.

    KNOWN DEVIATION!!! Metric columns are sorted alphabetically to keep
    the output deterministic. TDS section 9.2 does not require a
    particular ordering for them.
    """
    param_columns = [
        f"param.{parameter.name}" for parameter in contract.parameters
    ]

    metric_names: set[str] = set()

    for record in history:
        metric_names.update(record.metrics.keys())

    metric_columns = [
        f"metric.{name}" for name in sorted(metric_names)
    ]

    return [
        *_FIXED_COLUMNS,
        *param_columns,
        *metric_columns,
        "error_message",
    ]


def _flatten_record(
    record: TrialRecord,
    contract: OptimizationContract,
    fieldnames: Sequence[str],
) -> dict[str, object]:
    """Flatten one record with only its bounded diagnostic summary."""
    row: dict[str, object] = {
        "trial_id": record.trial_id,
        "execution_status": record.execution_status,
        "metrics_status": record.metrics_status,
        "execution_succeeded": record.execution_succeeded,
        "objective_eligible": is_eligible(record, contract),
        "runtime_seconds": record.runtime_seconds,
        "exit_code": record.exit_code,
        "timed_out": record.timed_out,
        "error_message": record.error_message,
    }

    for parameter in contract.parameters:
        # Every declared parameter should exist in every TrialRecord
        # A missing one means something went wrong earlier in the pipeline
        # Better to fail loudly than quietly write incorrect history
        row[f"param.{parameter.name}"] = record.parameters[parameter.name]

    for field in fieldnames:
        if field.startswith("metric."):
            metric_name = field[len("metric.") :]
            row[field] = record.metrics.get(metric_name, "")

    return row


def _atomic_write_text(destination: Path, content: str) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary_name, destination)
    except BaseException:
        _remove_temporary_file(temporary_name)
        raise


def _remove_temporary_file(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
