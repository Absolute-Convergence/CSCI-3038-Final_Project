"""Focused tests for the ZDT1 benchmark's math: the synthetic worker's ZDT1
formula and the hypervolume computation used to
score search results against it (compare_search_algorithms.py).

This deliberately does not re-run real subprocess trials -- that path is
already exercised for real by compare_search_algorithms.py itself. What
isn't covered anywhere else is whether the formulas are actually right,
which is what these tests check.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from black_box_optimizer.metrics import read_trial_metrics
from examples.zdt1_benchmark.compare_search_algorithms import (
    _quartiles,
    build_argument_parser,
    hypervolume_2d,
    mann_whitney_u,
    true_zdt1_hypervolume,
)
from hyperloop_workers.synthetic_worker import _NUM_VARIABLES, zdt1


_REPOSITORY = Path(__file__).resolve().parents[1]


class Zdt1FormulaTests(unittest.TestCase):
    """Verify the ZDT1 formula against known boundary cases and hand math."""

    def test_boundary_at_x1_zero_and_zero_nuisance(self) -> None:
        # g = 1 (every nuisance variable is 0), f1 = 0 -> f2 = g = 1.
        variables = [0.0] * _NUM_VARIABLES
        f1, f2 = zdt1(variables)
        self.assertEqual(f1, 0.0)
        self.assertEqual(f2, 1.0)

    def test_boundary_at_x1_one_and_zero_nuisance(self) -> None:
        # g = 1, f1 = 1 -> f2 = 1 * (1 - sqrt(1/1)) = 0. This is the single
        # best possible point in the whole search space.
        variables = [1.0] + [0.0] * (_NUM_VARIABLES - 1)
        f1, f2 = zdt1(variables)
        self.assertEqual(f1, 1.0)
        self.assertAlmostEqual(f2, 0.0)

    def test_f1_depends_only_on_the_first_variable(self) -> None:
        low_nuisance = [0.5] + [0.0] * (_NUM_VARIABLES - 1)
        high_nuisance = [0.5] + [1.0] * (_NUM_VARIABLES - 1)
        f1_low, _ = zdt1(low_nuisance)
        f1_high, _ = zdt1(high_nuisance)
        self.assertEqual(f1_low, f1_high)

    def test_larger_nuisance_values_make_f2_worse(self) -> None:
        low_nuisance = [0.5] + [0.0] * (_NUM_VARIABLES - 1)
        high_nuisance = [0.5] + [1.0] * (_NUM_VARIABLES - 1)
        _, f2_low = zdt1(low_nuisance)
        _, f2_high = zdt1(high_nuisance)
        self.assertLess(f2_low, f2_high)

    def test_matches_hand_computed_value(self) -> None:
        # Every nuisance value equal to the same constant c makes
        # g = 1 + 9c regardless of how many nuisance variables there are,
        # so this stays correct even if _NUM_VARIABLES changes later.
        nuisance_value = 0.3
        f1_input = 0.4
        variables = [f1_input] + [nuisance_value] * (_NUM_VARIABLES - 1)

        f1, f2 = zdt1(variables)

        expected_g = 1 + 9 * nuisance_value
        expected_f2 = expected_g * (1 - math.sqrt(f1_input / expected_g))
        self.assertAlmostEqual(f1, f1_input)
        self.assertAlmostEqual(f2, expected_f2)


class SyntheticWorkerProcessTests(unittest.TestCase):
    def test_distributed_module_runs_as_an_opaque_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hyperloop_workers.synthetic_worker",
                    "--x1",
                    "0.25",
                    "--x2",
                    "0",
                    "--x3",
                    "0",
                    "--x4",
                    "0",
                    "--metrics-out",
                    str(metrics_path),
                ],
                cwd=_REPOSITORY,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                read_trial_metrics(metrics_path),
                {"f1": 0.25, "f2": 0.5},
            )


class BenchmarkConfigurationTests(unittest.TestCase):
    def test_default_runs_ten_thousand_real_worker_trials(self) -> None:
        arguments = build_argument_parser().parse_args([])

        total = 2 * arguments.seeds * arguments.trials_per_run
        self.assertEqual(arguments.seeds, 10)
        self.assertEqual(arguments.trials_per_run, 500)
        self.assertEqual(total, 10_000)

    def test_seed_count_supports_one_five_and_ten_thousand(self) -> None:
        parser = build_argument_parser()

        for seeds, expected_total in ((1, 1_000), (5, 5_000), (10, 10_000)):
            with self.subTest(seeds=seeds):
                arguments = parser.parse_args(["--seeds", str(seeds)])
                total = 2 * arguments.seeds * arguments.trials_per_run
                self.assertEqual(total, expected_total)


class Hypervolume2DTests(unittest.TestCase):
    """Verify the dominated-area sweep against hand-worked geometry."""

    def test_hand_worked_three_point_staircase(self) -> None:
        # Rectangles: [0,0.5)x(1.1-1.0)=0.05, [0.5,1.0)x(1.1-0.5)=0.3,
        # [1.0,1.1]x(1.1-0.0)=0.11 -- total 0.46.
        points = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        result = hypervolume_2d(points, (1.1, 1.1))
        self.assertAlmostEqual(result, 0.46)

    def test_dominated_point_contributes_nothing(self) -> None:
        points = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.7, 0.9)]
        result = hypervolume_2d(points, (1.1, 1.1))
        self.assertAlmostEqual(result, 0.46)

    def test_empty_points_returns_zero(self) -> None:
        self.assertEqual(hypervolume_2d([], (1.1, 1.1)), 0.0)

    def test_point_at_or_beyond_reference_contributes_nothing(self) -> None:
        points = [(1.1, 0.5), (0.5, 1.1), (2.0, 2.0)]
        self.assertEqual(hypervolume_2d(points, (1.1, 1.1)), 0.0)

    def test_single_point_is_a_simple_rectangle(self) -> None:
        result = hypervolume_2d([(0.5, 0.5)], (1.0, 1.0))
        self.assertAlmostEqual(result, 0.25)

    def test_tied_second_objective_does_not_double_count(self) -> None:
        # Two points sharing the same y at different x -- only the first
        # (smaller x) should define a rectangle; the second adds nothing.
        points = [(0.0, 0.5), (0.3, 0.5)]
        result = hypervolume_2d(points, (1.0, 1.0))
        self.assertAlmostEqual(result, 0.5)


class TrueZdt1HypervolumeTests(unittest.TestCase):
    """Verify sampled true-front hypervolume against closed-form calculus."""

    def test_matches_closed_form_derivation(self) -> None:
        # For reference (1.1, 1.1), integrating the true front
        # f2 = 1 - sqrt(f1) over f1 in [0, 1] gives:
        #   integral_0^1 (1.1 - (1 - sqrt(f1))) df1 + 1.1 * (1.1 - 1)
        #   = (0.1 + 2/3) + 0.11 = 0.87667 (5 s.f.)
        expected = (0.1 + 2 / 3) + 0.11
        result = true_zdt1_hypervolume((1.1, 1.1), samples=20_000)
        self.assertAlmostEqual(result, expected, places=3)


class MannWhitneyUTests(unittest.TestCase):
    """Verify the hand-rolled Mann-Whitney U test (no scipy dependency)
    against hand-computable cases.

    Cross-checked interactively against real scipy.stats.mannwhitneyu
    during development (not a project dependency, so not imported here):
    the U statistic matched exactly in every case tried, and p-values
    matched closely once samples were realistically sized (~20 each, the
    scale a real comparison run actually uses). The two diverge slightly
    for tiny samples, where scipy defaults to an exact permutation test
    instead of the normal approximation used here -- expected and
    documented in mann_whitney_u()'s own docstring.
    """

    def test_fully_separated_samples_give_the_minimum_u_statistic(
        self,
    ) -> None:
        # Every value in the first sample is below every value in the
        # second, so it "wins" all 3*3=9 pairwise comparisons -- the
        # minimum possible U for it is 0, the strongest possible signal.
        u_statistic, p_value = mann_whitney_u([1, 2, 3], [4, 5, 6])
        self.assertEqual(u_statistic, 0.0)
        self.assertLess(p_value, 0.10)

    def test_ties_within_and_across_samples_average_correctly(self) -> None:
        # Hand-computed: combined sorted values are 1,1,2,2,3,3,3,4,4,5
        # (positions 1-10). The two 1's share ranks 1-2 (avg 1.5), the
        # two 2's share ranks 3-4 (avg 3.5), the three 3's share ranks
        # 5-7 (avg 6), the two 4's share ranks 8-9 (avg 8.5), the 5 is
        # rank 10. First sample [1,1,2,2,3] -> rank sum
        # 1.5+1.5+3.5+3.5+6 = 16. U = 16 - 5*6/2 = 16 - 15 = 1.
        u_statistic, _p_value = mann_whitney_u(
            [1, 1, 2, 2, 3], [3, 3, 4, 4, 5]
        )
        self.assertEqual(u_statistic, 1.0)

    def test_identical_distributions_are_not_significant(self) -> None:
        sample = [0.1 * i for i in range(1, 21)]
        _u_statistic, p_value = mann_whitney_u(sample, list(sample))
        self.assertEqual(p_value, 1.0)

    def test_clearly_separated_realistic_scale_samples_are_significant(
        self,
    ) -> None:
        # 20-per-group, non-overlapping ranges -- the scale and shape of
        # a real comparison run, where the normal approximation used
        # here is expected to be accurate.
        lower = [0.01 * i for i in range(1, 21)]
        higher = [0.01 * i + 0.5 for i in range(1, 21)]
        _u_statistic, p_value = mann_whitney_u(lower, higher)
        self.assertLess(p_value, 0.001)

    def test_p_value_is_symmetric_in_argument_order(self) -> None:
        sample_a = [1, 2, 3, 4, 5]
        sample_b = [2, 3, 4, 5, 6]
        _u_a, p_ab = mann_whitney_u(sample_a, sample_b)
        _u_b, p_ba = mann_whitney_u(sample_b, sample_a)
        self.assertAlmostEqual(p_ab, p_ba)


class QuartilesTests(unittest.TestCase):
    def test_single_value_has_no_spread(self) -> None:
        self.assertEqual(_quartiles([5.0]), (5.0, 5.0))

    def test_matches_hand_computed_quartiles(self) -> None:
        # statistics.quantiles([1,2,3,4], n=4) uses the exclusive method:
        # Q1=1.25, Q3=3.75.
        first_quartile, third_quartile = _quartiles([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(first_quartile, 1.25)
        self.assertAlmostEqual(third_quartile, 3.75)


if __name__ == "__main__":
    unittest.main()
