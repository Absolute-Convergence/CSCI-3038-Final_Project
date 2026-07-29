"""Focused tests for the one-row CSV metrics parser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from black_box_optimizer.metrics import read_trial_metrics


class MetricsParserTests(unittest.TestCase):
    """Verify parsing, validation, and read-only behavior."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)

    def write_csv(self, name: str, text: str) -> Path:
        """Create a temporary CSV file for an individual test case."""
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    # Tests that should successfully read valid metrics files.

    def test_reads_single_row_metrics(self) -> None:
        path = self.write_csv(
            "metrics.csv",
            "validation_accuracy,validation_loss\n0.91,0.42\n",
        )

        metrics = read_trial_metrics(path)

        self.assertEqual(metrics["validation_accuracy"], 0.91)
        self.assertEqual(metrics["validation_loss"], 0.42)

    def test_accepts_string_path(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\n1\n")

        metrics = read_trial_metrics(str(path))

        self.assertEqual(metrics["accuracy"], 1.0)

    def test_returned_mapping_is_read_only(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\n1\n")
        metrics = read_trial_metrics(path)

        with self.assertRaises(TypeError):
            metrics["accuracy"] = 2.0  # type: ignore[index]

    def test_header_names_are_case_sensitive(self) -> None:
        path = self.write_csv("metrics.csv", "Accuracy,accuracy\n1,2\n")

        metrics = read_trial_metrics(path)

        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(metrics["accuracy"], 2.0)

    # Tests that validate the overall structure of the CSV file.

    def test_missing_file_raises(self) -> None:
        missing = self.directory / "missing.csv"

        with self.assertRaises(FileNotFoundError):
            read_trial_metrics(missing)

    def test_empty_file_raises(self) -> None:
        path = self.write_csv("metrics.csv", "")

        with self.assertRaisesRegex(ValueError, "empty"):
            read_trial_metrics(path)

    def test_header_only_file_raises(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy,loss\n")

        with self.assertRaisesRegex(ValueError, "one data row"):
            read_trial_metrics(path)

    def test_extra_data_rows_raise(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\n1\n2\n")

        with self.assertRaisesRegex(ValueError, "exactly one data row"):
            read_trial_metrics(path)

    # Tests that validate the metric names.

    def test_mismatched_column_count_raises(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy,loss\n1\n")

        with self.assertRaisesRegex(ValueError, "lengths do not match"):
            read_trial_metrics(path)

    def test_duplicate_header_raises(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy,accuracy\n1,2\n")

        with self.assertRaisesRegex(ValueError, "unique"):
            read_trial_metrics(path)

    def test_blank_header_raises(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy, \n1,2\n")

        with self.assertRaisesRegex(ValueError, "blank"):
            read_trial_metrics(path)

    # Tests that validate individual metric values.

    def test_non_numeric_value_raises(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\nnot-a-number\n")

        with self.assertRaisesRegex(ValueError, "not numeric"):
            read_trial_metrics(path)

    def test_non_finite_value_raises(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\nnan\n")

        with self.assertRaisesRegex(ValueError, "finite"):
            read_trial_metrics(path)

    # Some programs save CSV files with a UTF-8 byte-order mark (BOM).
    # The parser should ignore it instead of treating it as part of the
    # first metric name.

    def test_utf8_bom_prefix_does_not_corrupt_first_header(self) -> None:
        path = self.directory / "metrics.csv"
        path.write_bytes("accuracy,loss\n0.9,0.1\n".encode("utf-8-sig"))

        metrics = read_trial_metrics(path)

        self.assertEqual(set(metrics), {"accuracy", "loss"})
        self.assertEqual(metrics["accuracy"], 0.9)

    # Metric names should be cleaned before they are returned so that
    # extra spaces in the CSV do not become dictionary keys.

    def test_header_whitespace_is_stripped_from_returned_keys(self) -> None:
        path = self.write_csv("metrics.csv", " accuracy , loss\n0.9,0.1\n")

        metrics = read_trial_metrics(path)

        self.assertEqual(set(metrics), {"accuracy", "loss"})


if __name__ == "__main__":
    unittest.main()
