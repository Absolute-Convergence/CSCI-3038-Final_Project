"""Authoritative result exports and explanatory Pareto visualization."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from black_box_optimizer.models import (
    ParameterKind,
    ProjectConfiguration,
)
from black_box_optimizer.pareto import is_eligible
from black_box_optimizer.persistence import RunDirectory
from black_box_optimizer.results import OptimizationResult


class ReportingError(RuntimeError):
    """Raised when a required report artifact cannot be completed."""


class ResultReporter(Protocol):
    """Reporting shape consumed by the lifecycle controller."""

    def write(self, result: OptimizationResult) -> None:
        ...


class Reporter:
    """Write authoritative run outputs without changing optimizer results."""

    def __init__(
        self,
        configuration: ProjectConfiguration,
        run_directory: RunDirectory,
    ) -> None:
        if not isinstance(configuration, ProjectConfiguration):
            raise TypeError("configuration must be a ProjectConfiguration")
        if not isinstance(run_directory, RunDirectory):
            raise TypeError("run_directory must be a RunDirectory")
        self._configuration = configuration
        self._run_directory = run_directory

    def write_resolved_configuration(self) -> None:
        """Write the validated JSON shape with resolved worker paths."""
        content = json.dumps(
            _configuration_document(self._configuration),
            indent=2,
            ensure_ascii=False,
        )
        _atomic_write_text(
            self._run_directory.path / "resolved_config.json",
            content + "\n",
        )

    def write(self, result: OptimizationResult) -> None:
        """Write final authoritative and explanatory result artifacts."""
        if not isinstance(result, OptimizationResult):
            raise TypeError("result must be an OptimizationResult")

        self.write_resolved_configuration()
        _atomic_write_text(
            self._run_directory.path / "pareto_front.csv",
            _pareto_csv(result, self._configuration),
        )
        _atomic_write_text(
            self._run_directory.path / "summary.txt",
            _summary(result, self._configuration),
        )
        _atomic_write_bytes(
            self._run_directory.path / "pareto_front.png",
            _pareto_plot(result, self._configuration),
        )


def _configuration_document(
    configuration: ProjectConfiguration,
) -> dict[str, object]:
    parameter_documents: list[dict[str, object]] = []
    for parameter in configuration.optimization.parameters:
        item: dict[str, object] = {
            "name": parameter.name,
            "kind": parameter.kind.value,
        }
        if parameter.kind is ParameterKind.CATEGORICAL:
            item["choices"] = list(parameter.choices)
        else:
            item["minimum"] = parameter.minimum
            item["maximum"] = parameter.maximum
        parameter_documents.append(item)

    return {
        "worker": {
            "command": list(configuration.worker.command),
            "metrics_argument": configuration.worker.metrics_argument,
            "timeout_seconds": configuration.worker.timeout_seconds,
        },
        "optimization": {
            "parameters": parameter_documents,
            "objectives": [
                {
                    "metric_name": objective.metric_name,
                    "direction": objective.direction.value,
                }
                for objective in configuration.optimization.objectives
            ],
        },
        "algorithm": {
            "name": configuration.algorithm.name,
            "seed": configuration.algorithm.seed,
        },
        "stop_policy": {
            "max_trials": configuration.stop_policy.max_trials,
        },
    }


def _pareto_csv(
    result: OptimizationResult,
    configuration: ProjectConfiguration,
) -> str:
    contract = configuration.optimization
    parameter_columns = [
        f"param.{parameter.name}" for parameter in contract.parameters
    ]
    objective_names = tuple(
        objective.metric_name for objective in contract.objectives
    )
    extra_metrics = sorted(
        {
            metric_name
            for record in result.pareto_front.records
            for metric_name in record.metrics
            if metric_name not in objective_names
        }
    )
    metric_columns = [
        *(f"metric.{name}" for name in objective_names),
        *(f"metric.{name}" for name in extra_metrics),
    ]
    fieldnames = ["trial_id", *parameter_columns, *metric_columns]

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    for record in result.pareto_front.records:
        row: dict[str, object] = {"trial_id": record.trial_id}
        for parameter in contract.parameters:
            row[f"param.{parameter.name}"] = record.parameters[parameter.name]
        for metric_name in (*objective_names, *extra_metrics):
            row[f"metric.{metric_name}"] = record.metrics.get(metric_name, "")
        writer.writerow(row)
    return stream.getvalue()


def _summary(
    result: OptimizationResult,
    configuration: ProjectConfiguration,
) -> str:
    objective_text = ", ".join(
        f"{objective.metric_name} ({objective.direction.value})"
        for objective in configuration.optimization.objectives
    )
    lines = (
        f"status: {result.status}",
        f"termination_reason: {result.termination_reason}",
        f"attempted_trials: {result.attempted_count}",
        f"successful_trials: {result.successful_count}",
        f"valid_metrics_trials: {result.valid_metrics_count}",
        f"pareto_trials: {result.pareto_count}",
        f"objectives: {objective_text}",
        "pareto_front: complete non-dominated set; no weighted winner",
    )
    return "\n".join(lines) + "\n"


def _pareto_plot(
    result: OptimizationResult,
    configuration: ProjectConfiguration,
) -> bytes:
    contract = configuration.optimization
    x_objective, y_objective = contract.objectives[:2]
    eligible = tuple(
        record
        for record in result.history
        if is_eligible(record, contract)
    )
    front_ids = {
        record.trial_id for record in result.pareto_front.records
    }
    other_records = tuple(
        record for record in eligible if record.trial_id not in front_ids
    )

    figure = Figure(figsize=(7.0, 5.0), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)

    if other_records:
        other_x = [
            record.metrics[x_objective.metric_name]
            for record in other_records
        ]
        other_y = [
            record.metrics[y_objective.metric_name]
            for record in other_records
        ]
        axes.scatter(
            other_x,
            other_y,
            color="#9aa0a6",
            alpha=0.75,
            label="eligible trials",
        )
    if result.pareto_front.records:
        axes.scatter(
            [
                record.metrics[x_objective.metric_name]
                for record in result.pareto_front.records
            ],
            [
                record.metrics[y_objective.metric_name]
                for record in result.pareto_front.records
            ],
            color="#1565c0",
            label="Pareto front",
        )
    if not eligible:
        axes.text(
            0.5,
            0.5,
            "No eligible trials",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )

    axes.set_xlabel(
        f"{x_objective.metric_name} ({x_objective.direction.value})"
    )
    axes.set_ylabel(
        f"{y_objective.metric_name} ({y_objective.direction.value})"
    )
    axes.set_title("Optimization trials and complete Pareto front")
    axes.grid(alpha=0.2)
    if eligible:
        axes.legend()
    figure.tight_layout()

    stream = io.BytesIO()
    canvas.print_png(stream)
    return stream.getvalue()


def _atomic_write_text(destination: Path, content: str) -> None:
    data = content.encode("utf-8")
    _atomic_write_bytes(destination, data)


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary_name, destination)
    except OSError as error:
        Path(temporary_name).unlink(missing_ok=True)
        raise ReportingError(
            f"Failed to write required report {destination.name}: {error}"
        ) from error
