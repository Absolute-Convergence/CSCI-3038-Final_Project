"""Focused tests for the Iris example worker."""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from black_box_optimizer.metrics import read_trial_metrics
from examples.iris_torch.worker import (
    IrisClassifier,
    load_iris_data,
    split_train_validation,
    train_and_evaluate,
    write_metrics,
)

_WORKER_PATH = (
    Path(__file__).parent.parent / "examples" / "iris_torch" / "worker.py"
)


class LoadIrisDataTests(unittest.TestCase):
    """Verify the bundled dataset loads with the expected shape."""

    def test_loads_150_rows(self) -> None:
        features, labels = load_iris_data()
        self.assertEqual(features.shape, (150, 4))
        self.assertEqual(labels.shape, (150,))

    def test_labels_are_the_three_expected_classes(self) -> None:
        _, labels = load_iris_data()
        self.assertEqual(set(labels.tolist()), {0, 1, 2})


class SplitTrainValidationTests(unittest.TestCase):
    """Verify the 80/20 split size and determinism."""

    def test_split_sizes(self) -> None:
        features, labels = load_iris_data()
        train_x, train_y, val_x, val_y = split_train_validation(
            features, labels
        )
        self.assertEqual(len(train_x), 120)
        self.assertEqual(len(val_x), 30)
        self.assertEqual(len(train_y), 120)
        self.assertEqual(len(val_y), 30)

    def test_split_is_deterministic(self) -> None:
        features, labels = load_iris_data()
        first = split_train_validation(features, labels)
        second = split_train_validation(features, labels)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))


class IrisClassifierTests(unittest.TestCase):
    """Verify the network accepts different hidden sizes and returns
    three class scores."""

    def test_output_shape_matches_batch_and_class_count(self) -> None:
        model = IrisClassifier(hidden_size=16)
        batch = torch.zeros((5, 4))
        output = model(batch)
        self.assertEqual(output.shape, (5, 3))

    def test_small_and_large_hidden_size_both_work(self) -> None:
        for hidden_size in (4, 128):
            model = IrisClassifier(hidden_size=hidden_size)
            output = model(torch.zeros((1, 4)))
            self.assertEqual(output.shape, (1, 3))


class TrainAndEvaluateTests(unittest.TestCase):
    """Verify training produces valid, deterministic metrics."""

    def test_accuracy_and_loss_are_in_valid_ranges(self) -> None:
        accuracy, loss = train_and_evaluate(
            learning_rate=0.05, hidden_size=16, epochs=5, batch_size=16
        )
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertTrue(math.isfinite(accuracy))
        self.assertGreater(loss, 0.0)
        self.assertTrue(math.isfinite(loss))

    def test_same_hyperparameters_produce_identical_results(self) -> None:
        first = train_and_evaluate(
            learning_rate=0.05, hidden_size=16, epochs=5, batch_size=16
        )
        second = train_and_evaluate(
            learning_rate=0.05, hidden_size=16, epochs=5, batch_size=16
        )
        self.assertEqual(first, second)

    def test_different_hyperparameters_are_accepted(self) -> None:
        # Exercises every batch_size choice and the min/max hidden_size
        # bounds from iris_config.json, not just one arbitrary combination.
        for hidden_size in (4, 128):
            for batch_size in (8, 16, 32):
                accuracy, loss = train_and_evaluate(
                    learning_rate=0.01,
                    hidden_size=hidden_size,
                    epochs=2,
                    batch_size=batch_size,
                )
                self.assertGreaterEqual(accuracy, 0.0)
                self.assertLessEqual(accuracy, 1.0)
                self.assertTrue(math.isfinite(loss))


class WriteMetricsTests(unittest.TestCase):
    """Verify the written CSV is actually readable by metrics.py."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)

    def test_written_metrics_parse_through_real_metrics_py(self) -> None:
        path = self.directory / "metrics.csv"
        write_metrics(path, accuracy=0.9, loss=0.2, training_time_seconds=1.5)

        metrics = read_trial_metrics(path)

        self.assertEqual(metrics["validation_accuracy"], 0.9)
        self.assertEqual(metrics["validation_loss"], 0.2)
        self.assertEqual(metrics["training_time_seconds"], 1.5)


class WorkerCliTests(unittest.TestCase):
    """Verify the worker runs correctly as a real subprocess."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)

    def test_full_subprocess_invocation(self) -> None:
        metrics_path = self.directory / "metrics.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(_WORKER_PATH),
                "--learning-rate", "0.05",
                "--hidden-size", "8",
                "--epochs", "3",
                "--batch-size", "8",
                "--metrics-out", str(metrics_path),
            ],
            shell=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        metrics = read_trial_metrics(metrics_path)
        self.assertIn("validation_accuracy", metrics)
        self.assertIn("validation_loss", metrics)
        self.assertIn("training_time_seconds", metrics)


if __name__ == "__main__":
    unittest.main()
