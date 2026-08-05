"""Authoritative result exports and explanatory Pareto visualization."""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from black_box_optimizer.models import (
    Direction,
    Objective,
    ParameterKind,
    ProjectConfiguration,
)
from black_box_optimizer.pareto import is_eligible
from black_box_optimizer.persistence import RunDirectory
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.results import OptimizationResult


class ReportingError(RuntimeError):
    """Raised when a required report artifact cannot be completed."""


class ResultReporter(Protocol):
    """Reporting shape consumed by the lifecycle controller."""

    def write(self, result: OptimizationResult) -> None:
        ...


class Reporter:
    """Write authoritative outputs with per-file atomic replacement."""

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
        """Write final artifacts; the group itself is not transactional."""
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


# Chart palette -- dataviz skill's validated default instance
# (references/palette.md), "highlight one, gray the rest" pattern: the
# Pareto front is the one series that matters, so it gets the vivid,
# validated categorical hue while every other eligible trial recedes
# into muted context.
_SURFACE = "#fcfcfb"
_GRIDLINE = "#e1e0d9"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_FRONT_COLOR = "#eb6834"


def _label_extreme_points(
    axes,
    pareto_records: Sequence[TrialRecord],
    x_objective: Objective,
    y_objective: Objective,
) -> None:
    """Label the best-per-objective trial on the Pareto front with its
    trial_id, instead of every point -- a front can easily have dozens
    of points (we've seen 50-70 in real runs), where labeling all of
    them would just be visual noise. These are also the same extreme
    values the CLI's own "Best <objective>" summary already reports, so
    the chart stays consistent with what the terminal already told the
    user.
    """
    for objective, axis in ((x_objective, "x"), (y_objective, "y")):
        best = (
            max
            if objective.direction == Direction.MAXIMIZE
            else min
        )
        best_record = best(
            pareto_records,
            key=lambda record: record.metrics[objective.metric_name],
        )
        point = (
            best_record.metrics[x_objective.metric_name],
            best_record.metrics[y_objective.metric_name],
        )
        offset = (6, 6) if axis == "x" else (6, -10)
        axes.annotate(
            f"trial {best_record.trial_id}",
            point,
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            color=_FRONT_COLOR,
            fontweight="bold",
        )


_CHART_TITLE = "Optimization trials and complete Pareto front"


def _grid_shape(pair_count: int) -> tuple[int, int]:
    """Return (rows, columns) for a roughly square subplot grid."""
    columns = math.ceil(math.sqrt(pair_count))
    rows = math.ceil(pair_count / columns)
    return rows, columns


def _plot_objective_pair(
    axes,
    x_objective: Objective,
    y_objective: Objective,
    other_records: Sequence[TrialRecord],
    pareto_records: Sequence[TrialRecord],
) -> None:
    """Draw one x/y projection of the trials onto ``axes``.

    Every objective pair shares this exact drawing logic so a two-
    objective run and one axes of a many-objective grid look identical.
    """
    if other_records:
        axes.scatter(
            [
                record.metrics[x_objective.metric_name]
                for record in other_records
            ],
            [
                record.metrics[y_objective.metric_name]
                for record in other_records
            ],
            color=_MUTED_INK,
            alpha=0.55,
            s=28,
            linewidths=0,
            label="eligible trials",
        )
    axes.scatter(
        [
            record.metrics[x_objective.metric_name]
            for record in pareto_records
        ],
        [
            record.metrics[y_objective.metric_name]
            for record in pareto_records
        ],
        color=_FRONT_COLOR,
        s=46,
        edgecolors=_SURFACE,
        linewidths=1.0,
        zorder=3,
        label="Pareto front",
    )
    _label_extreme_points(axes, pareto_records, x_objective, y_objective)

    axes.set_xlabel(
        f"{x_objective.metric_name} ({x_objective.direction.value})",
        color=_SECONDARY_INK,
    )
    axes.set_ylabel(
        f"{y_objective.metric_name} ({y_objective.direction.value})",
        color=_SECONDARY_INK,
    )
    axes.set_facecolor(_SURFACE)
    axes.grid(color=_GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(_GRIDLINE)
    axes.tick_params(colors=_SECONDARY_INK)


def _empty_pareto_plot() -> bytes:
    figure = Figure(figsize=(7.0, 5.0), dpi=120, facecolor=_SURFACE)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    axes.set_facecolor(_SURFACE)
    axes.text(
        0.5,
        0.5,
        "No eligible trials",
        ha="center",
        va="center",
        color=_SECONDARY_INK,
        transform=axes.transAxes,
    )
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(_GRIDLINE)
    axes.set_title(_CHART_TITLE, color=_PRIMARY_INK, fontweight="bold")
    figure.tight_layout()

    stream = io.BytesIO()
    canvas.print_png(stream)
    return stream.getvalue()


def _pareto_plot(
    result: OptimizationResult,
    configuration: ProjectConfiguration,
) -> bytes:
    """Chart every pairwise projection of the Pareto front.

    Two objectives need exactly one x/y plot. Three or more need one
    plot per unique pair (a 4-objective run has 6, a 5-objective run has
    10) laid out in a grid, since there is no single 2D view that shows
    a front with more than two dimensions at once.
    """
    contract = configuration.optimization
    eligible = tuple(
        record
        for record in result.history
        if is_eligible(record, contract)
    )
    if not eligible:
        return _empty_pareto_plot()

    front_ids = {
        record.trial_id for record in result.pareto_front.records
    }
    other_records = tuple(
        record for record in eligible if record.trial_id not in front_ids
    )
    pairs = tuple(itertools.combinations(contract.objectives, 2))
    rows, columns = _grid_shape(len(pairs))
    width = max(7.0, columns * 4.0)
    height = max(5.0, rows * 3.6)

    figure = Figure(figsize=(width, height), dpi=120, facecolor=_SURFACE)
    canvas = FigureCanvasAgg(figure)
    for index, (x_objective, y_objective) in enumerate(pairs, start=1):
        axes = figure.add_subplot(rows, columns, index)
        _plot_objective_pair(
            axes,
            x_objective,
            y_objective,
            other_records,
            result.pareto_front.records,
        )

    legend_kwargs = {
        "frameon": True,
        "facecolor": _SURFACE,
        "edgecolor": _GRIDLINE,
        "labelcolor": _SECONDARY_INK,
    }
    if len(pairs) == 1:
        figure.axes[0].set_title(
            _CHART_TITLE, color=_PRIMARY_INK, fontweight="bold"
        )
        figure.axes[0].legend(**legend_kwargs)
        figure.tight_layout()
    else:
        # Title and legend are both centered and stacked, not sharing a
        # corner -- a figure.legend(loc="upper right") collided with the
        # suptitle on narrower grids (few pairs -> few columns leaves
        # little horizontal room to separate them).
        handles, labels = figure.axes[0].get_legend_handles_labels()
        figure.suptitle(
            _CHART_TITLE, y=0.99, color=_PRIMARY_INK, fontweight="bold"
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            **legend_kwargs,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

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
    except Exception as error:
        Path(temporary_name).unlink(missing_ok=True)
        raise ReportingError(
            f"Failed to write required report {destination.name}: {error}"
        ) from error
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
