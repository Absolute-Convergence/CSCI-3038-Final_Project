"""
metrics.py

This file reads the metrics produced by only a single completed worker trial.
It will verify that the metrics file follows the expected format, convert each
metric into a usable number, reject any values that would cause trouble later,
and finally return the results in a read-only mapping for the rest of the
project to utilize.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


class MetricsFormatError(ValueError):
    """The metrics file exists, but something inside it is shaped wrong."""


class NonFiniteMetricError(ValueError):
    """A metric became a float, but unfortunately it was NaN or infinite."""


def read_trial_metrics(metrics_path: str | Path) -> Mapping[str, float]:
    """
    Read the metrics file for one completed trial.

    This function opens the CSV file, validates its structure,
    converts each metric value into a float, makes sure those
    values are actually usable, and returns the finished results
    as a read-only mapping.
    """
    path = Path(metrics_path)

    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")

    # Reads every non-empty row from the CSV file
    # Each trial should create one header row and one row of values
    with path.open(newline="", encoding="utf-8-sig") as infile:
        rows = [row for row in csv.reader(infile) if row]

    # Must be sure there is actually data in the file
    if not rows:
        raise MetricsFormatError("metrics file is empty")

    if len(rows) == 1:
        raise MetricsFormatError(
            "metrics file must contain a header row and one data row"
        )

    if len(rows) > 2:
        raise MetricsFormatError(
            "metrics file must contain exactly one data row"
        )

    # Now we separate the metric names from the metric values
    header, data = rows

    # This removes extra spaces and also validates metric names
    header = _clean_headers(header)

    # Each metric name needs to have a single matching value
    if len(header) != len(data):
        raise MetricsFormatError(
            "metrics header and data row lengths do not match"
        )

    metrics = {}

    # She will convert each value into a float, make sure it is valid,
    # and store it by metric name
    for name, raw_value in zip(header, data):
        metrics[name] = _parse_metric_value(name, raw_value)

    # Return a read-only mapping so the parsed results cannot
    # accidentally be modified later
    return MappingProxyType(metrics)


def _clean_headers(header: list[str]) -> tuple[str, ...]:
    """
    Clean and validate the metric names from the header row.

    Leading and trailing whitespace is removed before checking
    that every metric has a name and that no metric name appears
    more than once.
    """
    if not header:
        raise MetricsFormatError("metrics header row cannot be empty")

    # This will remove any extra spaces around the metric names
    cleaned = tuple(name.strip() for name in header)

    # Metric names cannot be blank!
    if any(not name for name in cleaned):
        raise MetricsFormatError("metrics header names cannot be blank")

    # Every metric name should be unique!
    if len(set(cleaned)) != len(cleaned):
        raise MetricsFormatError("metrics header names must be unique")

    return cleaned


def _parse_metric_value(name: str, raw_value: str) -> float:
    """
    Convert one metric value from text into a usable number.

    The value must successfully become a float, but it also has
    to be finite. NaN and infinity technically count as floats,
    but they would make comparisons and sorting behave strangely,
    so they are absolutely not invited.
    """
    # Attempt to convert the text from the CSV into a float
    # If it cannot, we will raise a more useful metrics-specific error
    try:
        value = float(raw_value)
    except ValueError as error:
        raise MetricsFormatError(
            f"metric {name!r} value is not numeric: {raw_value!r}"
        ) from error

    # Reject NaN and infinity because they are totally invalid metric values!
    if not math.isfinite(value):
        raise NonFiniteMetricError(
            f"metric {name!r} value must be finite: {value}"
        )

    return value