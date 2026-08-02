"""Tiny opaque worker used by CLI acceptance tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.metrics_out.open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("score", "cost"))
        writer.writerow((float(arguments.x), float(arguments.x)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
