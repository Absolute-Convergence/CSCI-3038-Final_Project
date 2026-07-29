"""
metrics.py

This file reads the metrics produced by only a single completed worker trial.
It will verify that the metrics file follows the expected format, then converts
each metric into a usable number, and finally returns the results in a read-only
mapping for the rest of the project to utilize.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


def read_trial_metrics(metrics_path: str | Path) -> Mapping[str, float]:
    """
    Read the metrics file for one completed trial.

    This function opens the CSV file, validates its structure,
    converts each metric value into a float, and returns the
    finished results as a read-only mapping.
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
        raise ValueError("metrics file is empty")

    if len(rows) == 1:
        raise ValueError(
            "metrics file must contain a header row and one data row"
        )

    if len(rows) > 2:
        raise ValueError("metrics file must contain exactly one data row")

    # Now we separate the metric names from the metric values
    header, data = rows

    # This removes extra spaces and also validates metric names
    header = _clean_headers(header)

    # Each metric name needs to have a single matching value
    if len(header) != len(data):
        raise ValueError("metrics header and data row lengths do not match")

    metrics = {}

    # She will convert each value into a float and store it by metric name
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
        raise ValueError("metrics header row cannot be empty")

    # This will remove any extra spaces around the metric names
    cleaned = tuple(name.strip() for name in header)

    # Metric names cannot be blank!
    if any(not name for name in cleaned):
        raise ValueError("metrics header names cannot be blank")

    # Every metric name should be unique!
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("metrics header names must be unique")

    return cleaned


def _parse_metric_value(name: str, raw_value: str) -> float:
    """
    Convert one metric value from text into a usable number.

    The value must be numeric and finite so it can safely be
    compared, sorted, and analyzed later in the program.
    """
    # Attempt to convert the text from the CSV into a float
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"metric {name!r} value is not numeric: {raw_value!r}"
        ) from error

    # Reject NaN and infinity because they are totally invalid metric values!
    if not math.isfinite(value):
        raise ValueError(f"metric {name!r} value must be finite: {value}")

    return value
