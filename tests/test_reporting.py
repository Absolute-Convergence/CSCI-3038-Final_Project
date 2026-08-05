"""Focused tests for authoritative result reporting."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from black_box_optimizer.models import (
    AlgorithmSpec,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    ProjectConfiguration,
    StopPolicy,
    WorkerSpec,
)
from black_box_optimizer.persistence import create_run_directory
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.reporting import (
    Reporter,
    ReportingError,
    _grid_shape,
    _label_extreme_points,
    _pareto_plot,
)
from black_box_optimizer.results import build_optimization_result


def make_configuration() -> ProjectConfiguration:
    return ProjectConfiguration(
        worker=WorkerSpec(
            command=("C:\\resolved\\python.exe", "C:\\project\\worker.py"),
            metrics_argument="--metrics-out",
            timeout_seconds=30.0,
        ),
        optimization=OptimizationContract(
            parameters=(
                ParameterDefinition("x", ParameterKind.INTEGER, 0, 10),
                ParameterDefinition(
                    "mode",
                    ParameterKind.CATEGORICAL,
                    choices=("fast", "careful"),
                ),
            ),
            objectives=(
                Objective("score", Direction.MAXIMIZE),
                Objective("cost", Direction.MINIMIZE),
                Objective("time", Direction.MINIMIZE),
            ),
        ),
        algorithm=AlgorithmSpec("random_search", seed=7),
        stop_policy=StopPolicy(max_trials=2),
    )


def make_record(
    trial_id: int,
    *,
    score: float,
    cost: float,
    time_value: float,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parameters={"x": trial_id, "mode": "fast"},
        metrics={
            "score": score,
            "cost": cost,
            "time": time_value,
            "z_extra": float(trial_id),
        },
        execution_status="completed",
        metrics_status="valid",
        runtime_seconds=0.2,
        exit_code=0,
        timed_out=False,
    )


class ReporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.configuration = make_configuration()
        self.run_directory = create_run_directory(self._temporary.name)
        self.reporter = Reporter(self.configuration, self.run_directory)

    def test_resolved_configuration_preserves_shape_and_order(self) -> None:
        self.reporter.write_resolved_configuration()

        path = self.run_directory.path / "resolved_config.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            list(document),
            ["worker", "optimization", "algorithm", "stop_policy"],
        )
        self.assertEqual(
            document["worker"]["command"],
            ["C:\\resolved\\python.exe", "C:\\project\\worker.py"],
        )
        self.assertEqual(
            [
                item["name"]
                for item in document["optimization"]["parameters"]
            ],
            ["x", "mode"],
        )

    def test_write_creates_all_final_artifacts(self) -> None:
        first = make_record(0, score=0.7, cost=0.4, time_value=1.0)
        second = make_record(1, score=0.9, cost=0.2, time_value=0.8)
        result = build_optimization_result(
            (first, second),
            self.configuration.optimization,
            "maximum_trials",
        )

        self.reporter.write(result)

        expected = {
            "resolved_config.json",
            "pareto_front.csv",
            "summary.txt",
            "pareto_front.png",
            "trials",
        }
        self.assertEqual(
            {path.name for path in self.run_directory.path.iterdir()},
            expected,
        )
        image = (self.run_directory.path / "pareto_front.png").read_bytes()
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_pareto_csv_has_deterministic_columns_and_front_order(self) -> None:
        first = make_record(0, score=0.9, cost=0.5, time_value=0.8)
        second = make_record(1, score=0.8, cost=0.2, time_value=0.6)
        result = build_optimization_result(
            (first, second),
            self.configuration.optimization,
            "maximum_trials",
        )

        self.reporter.write(result)

        with (self.run_directory.path / "pareto_front.csv").open(
            newline="",
            encoding="utf-8",
        ) as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)

        self.assertEqual(
            reader.fieldnames,
            [
                "trial_id",
                "param.x",
                "param.mode",
                "metric.score",
                "metric.cost",
                "metric.time",
                "metric.z_extra",
            ],
        )
        self.assertEqual([row["trial_id"] for row in rows], ["0", "1"])

    def test_summary_has_status_counts_and_no_winner(self) -> None:
        record = make_record(0, score=0.9, cost=0.2, time_value=0.8)
        result = build_optimization_result(
            (record,),
            self.configuration.optimization,
            "search_exhausted",
        )

        self.reporter.write(result)

        summary = (self.run_directory.path / "summary.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("status: completed", summary)
        self.assertIn("attempted_trials: 1", summary)
        self.assertIn("pareto_trials: 1", summary)
        self.assertIn("no weighted winner", summary)

    def test_empty_front_still_writes_header_and_plot(self) -> None:
        result = build_optimization_result(
            (),
            self.configuration.optimization,
            "search_exhausted",
        )

        self.reporter.write(result)

        csv_lines = (
            self.run_directory.path / "pareto_front.csv"
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(csv_lines), 1)
        self.assertIn("metric.score", csv_lines[0])
        self.assertGreater(
            (self.run_directory.path / "pareto_front.png").stat().st_size,
            0,
        )

    def test_failed_atomic_replace_retains_previous_summary(self) -> None:
        record = make_record(0, score=0.9, cost=0.2, time_value=0.8)
        result = build_optimization_result(
            (record,),
            self.configuration.optimization,
            "maximum_trials",
        )
        self.reporter.write(result)
        summary_path = self.run_directory.path / "summary.txt"
        previous = summary_path.read_bytes()
        real_replace = os.replace

        def fail_summary(source, destination):
            if Path(destination).name == "summary.txt":
                raise OSError("disk unavailable")
            return real_replace(source, destination)

        with patch(
            "black_box_optimizer.reporting.os.replace",
            side_effect=fail_summary,
        ):
            with self.assertRaisesRegex(ReportingError, "summary.txt"):
                self.reporter.write(result)

        self.assertEqual(summary_path.read_bytes(), previous)
        self.assertEqual(
            tuple(self.run_directory.path.glob("summary.txt.*.tmp")),
            (),
        )

    def test_report_group_is_explicitly_not_transactional(self) -> None:
        record = make_record(0, score=0.9, cost=0.2, time_value=0.8)
        result = build_optimization_result(
            (record,),
            self.configuration.optimization,
            "maximum_trials",
        )
        real_replace = os.replace

        def fail_summary(source, destination):
            if Path(destination).name == "summary.txt":
                raise OSError("disk unavailable")
            return real_replace(source, destination)

        with patch(
            "black_box_optimizer.reporting.os.replace",
            side_effect=fail_summary,
        ):
            with self.assertRaises(ReportingError):
                self.reporter.write(result)

        self.assertTrue(
            (self.run_directory.path / "resolved_config.json").is_file()
        )
        self.assertTrue(
            (self.run_directory.path / "pareto_front.csv").is_file()
        )
        self.assertFalse(
            (self.run_directory.path / "summary.txt").exists()
        )
        self.assertFalse(
            (self.run_directory.path / "pareto_front.png").exists()
        )

    def test_constructor_rejects_a_non_project_configuration(self) -> None:
        with self.assertRaisesRegex(TypeError, "ProjectConfiguration"):
            Reporter("not a configuration", self.run_directory)

    def test_constructor_rejects_a_non_run_directory(self) -> None:
        with self.assertRaisesRegex(TypeError, "RunDirectory"):
            Reporter(self.configuration, "not a run directory")

    def test_write_rejects_a_non_optimization_result(self) -> None:
        with self.assertRaisesRegex(TypeError, "OptimizationResult"):
            self.reporter.write("not a result")


def make_point_record(
    trial_id: int, *, x: float, y: float, x_name: str, y_name: str
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parameters={},
        metrics={x_name: x, y_name: y},
        execution_status="completed",
        metrics_status="valid",
        runtime_seconds=0.1,
        exit_code=0,
        timed_out=False,
    )


class LabelExtremePointsTests(unittest.TestCase):
    x_objective = Objective("distance", Direction.MAXIMIZE)
    y_objective = Objective("accuracy", Direction.MINIMIZE)

    def label_calls(self, records: tuple[TrialRecord, ...]) -> list[str]:
        axes = Mock()
        _label_extreme_points(
            axes, records, self.x_objective, self.y_objective
        )
        return [call.args[0] for call in axes.annotate.call_args_list]

    def make_record(self, trial_id: int, *, x: float, y: float) -> TrialRecord:
        return make_point_record(
            trial_id, x=x, y=y, x_name="distance", y_name="accuracy"
        )

    def test_labels_the_best_trial_per_axis_by_direction(self) -> None:
        records = (
            self.make_record(1, x=10.0, y=0.5),
            self.make_record(2, x=30.0, y=0.9),  # best (max) distance
            self.make_record(3, x=20.0, y=0.1),  # best (min) accuracy
        )

        labels = self.label_calls(records)

        self.assertEqual(sorted(labels), ["trial 2", "trial 3"])

    def test_a_tie_on_one_axis_deterministically_picks_the_first_record(
        self,
    ) -> None:
        # Both records tie for the best (max) distance, but trial 6 is
        # also the legitimately best (min) accuracy -- so the two axis
        # labels must be checked independently, in call order (x, y),
        # rather than by membership: "trial 6" belongs in the output,
        # just as the *second* (y-axis) label, not the first.
        records = (
            self.make_record(5, x=30.0, y=0.9),
            self.make_record(6, x=30.0, y=0.2),
        )

        axes = Mock()
        _label_extreme_points(
            axes, records, self.x_objective, self.y_objective
        )
        x_label, y_label = (
            call.args[0] for call in axes.annotate.call_args_list
        )

        # Python's max() keeps the first-encountered item on a tie, so
        # trial 5 -- not trial 6 -- must win the x-axis (distance) label.
        self.assertEqual(x_label, "trial 5")
        self.assertEqual(y_label, "trial 6")

    def test_a_single_record_front_labels_the_same_point_on_both_axes(
        self,
    ) -> None:
        records = (self.make_record(9, x=15.0, y=0.5),)

        labels = self.label_calls(records)

        self.assertEqual(labels, ["trial 9", "trial 9"])


class GridShapeTests(unittest.TestCase):
    def test_two_objectives_need_exactly_one_plot(self) -> None:
        # C(2, 2) == 1 pair
        self.assertEqual(_grid_shape(1), (1, 1))

    def test_three_objectives_need_a_grid_that_fits_all_three_pairs(
        self,
    ) -> None:
        # C(3, 2) == 3 pairs
        rows, columns = _grid_shape(3)
        self.assertGreaterEqual(rows * columns, 3)

    def test_five_objectives_need_a_roughly_square_ten_plot_grid(
        self,
    ) -> None:
        # C(5, 2) == 10 pairs
        rows, columns = _grid_shape(10)
        self.assertGreaterEqual(rows * columns, 10)
        self.assertLessEqual(abs(rows - columns), 1)


class ParetoPlotMultiObjectiveTests(unittest.TestCase):
    def make_configuration(self, objective_count: int) -> ProjectConfiguration:
        objectives = tuple(
            Objective(f"objective_{index}", Direction.MAXIMIZE)
            for index in range(objective_count)
        )
        return ProjectConfiguration(
            worker=WorkerSpec(
                command=("python3", "worker.py"),
                metrics_argument="--metrics-out",
                timeout_seconds=30.0,
            ),
            optimization=OptimizationContract(
                parameters=(
                    ParameterDefinition("x", ParameterKind.FLOAT, 0.0, 1.0),
                ),
                objectives=objectives,
            ),
            algorithm=AlgorithmSpec("random_search", seed=1),
            stop_policy=StopPolicy(max_trials=3),
        )

    def make_record(
        self, trial_id: int, objective_count: int
    ) -> TrialRecord:
        return TrialRecord(
            trial_id=trial_id,
            parameters={"x": 0.5},
            metrics={
                f"objective_{index}": float(trial_id + index)
                for index in range(objective_count)
            },
            execution_status="completed",
            metrics_status="valid",
            runtime_seconds=0.1,
            exit_code=0,
            timed_out=False,
        )

    def test_four_objectives_render_without_error(self) -> None:
        configuration = self.make_configuration(4)
        records = tuple(self.make_record(i, 4) for i in range(3))
        result = build_optimization_result(
            records, configuration.optimization, "maximum_trials"
        )

        image = _pareto_plot(result, configuration)

        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
