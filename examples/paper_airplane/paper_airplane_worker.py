"""
paper_airplane_worker.py

The Paper Airplane Design Problem (PADP-5): a fun-themed opaque worker
with a real, mathematically known Pareto front, for demos.

Five design parameters (wing_span_cm, nose_weight_g, wing_angle_deg,
fold_sharpness, paper_thickness_gsm) map onto two objectives to
maximize: flight_distance_m and landing_accuracy_pct.

Same two-tier construction ZDT1 uses (one "primary" variable, the rest
averaged into a "nuisance" penalty), but built directly for a genuine
maximize/maximize trade-off instead of reusing ZDT1's own minimize/
minimize formula verbatim -- an earlier version of this file tried
reusing ZDT1's f2 = g*(1 - sqrt(x1/g)) directly and it silently broke:
f2 always decreases as x1 increases (regardless of g) *and* always
increases as g increases, so no single monotonic transform of f2 could
make "more distance" cost accuracy while also making "worse
craftsmanship" cost accuracy -- one of those two properties always came
out backwards. Verified by hand before writing this version; see the
project's decision doc for the numbers.

    x1            = normalized wing_span_cm  (the "primary" variable)
    nuisance_mean = mean of the other 4 normalized params
                    (0 = best craftsmanship)

    flight_distance_m    = x1 * 30.0
    landing_accuracy_pct = 100 * (1 - x1) * (1 - nuisance_mean)

The true optimal front (best possible landing_accuracy_pct for a given
flight_distance_m) happens at nuisance_mean=0 -- every nuisance
parameter at its best setting (zero weight, zero angle, perfectly sharp
folds, lightest paper) -- where accuracy_pct = 100 - (10/3)*distance_m,
a straight line from (0m, 100%) to (30m, 0%): a real trade-off, bigger
wings fly farther but are harder to land precisely. Worse nuisance
settings drag accuracy down further from that line, down to 0% in the
worst case. Verified computationally that no off-front point ever
dominates a front point.

IMPORTANT: this file intentionally lives outside black_box_optimizer.
The optimizer never imports it -- as far as the optimizer is concerned
this is just an opaque subprocess that accepts parameters and writes
metrics, same contract as iris_worker.py and synthetic_worker.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_WING_SPAN_RANGE = (5.0, 20.0)      # cm
_NOSE_WEIGHT_RANGE = (0.0, 5.0)      # grams
_WING_ANGLE_RANGE = (0.0, 45.0)      # degrees
_PAPER_THICKNESS_RANGE = (60.0, 160.0)  # gsm

_MAX_DISTANCE_M = 30.0


def _normalize(value: float, low: float, high: float) -> float:
    return (value - low) / (high - low)


def design_paper_airplane(
    wing_span_cm: float,
    nose_weight_g: float,
    wing_angle_deg: float,
    fold_sharpness: float,
    paper_thickness_gsm: float,
) -> tuple[float, float]:
    """Evaluate one paper airplane design.

    Returns (flight_distance_m, landing_accuracy_pct), both maximized.
    """
    x1 = _normalize(wing_span_cm, *_WING_SPAN_RANGE)
    nuisance_values = (
        _normalize(nose_weight_g, *_NOSE_WEIGHT_RANGE),
        _normalize(wing_angle_deg, *_WING_ANGLE_RANGE),
        fold_sharpness,  # already a 0-1 crispness score
        _normalize(paper_thickness_gsm, *_PAPER_THICKNESS_RANGE),
    )
    nuisance_mean = sum(nuisance_values) / len(nuisance_values)

    flight_distance_m = x1 * _MAX_DISTANCE_M
    landing_accuracy_pct = 100.0 * (1.0 - x1) * (1.0 - nuisance_mean)

    return flight_distance_m, landing_accuracy_pct


def write_metrics(
    metrics_path: Path,
    flight_distance_m: float,
    landing_accuracy_pct: float,
) -> None:
    """Write one completed trial in the CSV format expected by metrics.py"""
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flight_distance_m", "landing_accuracy_pct"])
        writer.writerow([flight_distance_m, landing_accuracy_pct])


def main() -> None:
    """Read the trial parameters, evaluate the design, and write metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wing-span-cm", type=float, required=True)
    parser.add_argument("--nose-weight-g", type=float, required=True)
    parser.add_argument("--wing-angle-deg", type=float, required=True)
    parser.add_argument("--fold-sharpness", type=float, required=True)
    parser.add_argument("--paper-thickness-gsm", type=float, required=True)
    parser.add_argument("--metrics-out", type=str, required=True)
    args = parser.parse_args()

    flight_distance_m, landing_accuracy_pct = design_paper_airplane(
        args.wing_span_cm,
        args.nose_weight_g,
        args.wing_angle_deg,
        args.fold_sharpness,
        args.paper_thickness_gsm,
    )

    write_metrics(
        Path(args.metrics_out),
        flight_distance_m,
        landing_accuracy_pct,
    )


if __name__ == "__main__":
    main()
