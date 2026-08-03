"""Focused tests for the ZDT1 benchmark's math: the ZDT1 formula itself
(examples/zdt1_benchmark/worker.py) and the hypervolume computation used to
score search results against it (compare_search_algorithms.py).

This deliberately does not re-run real subprocess trials -- that path is
already exercised for real by compare_search_algorithms.py itself. What
isn't covered anywhere else is whether the formulas are actually right,
which is what these tests check.
"""

from __future__ import annotations

import math
import unittest

from examples.zdt1_benchmark.compare_search_algorithms import (
    hypervolume_2d,
    true_zdt1_hypervolume,
)
from examples.zdt1_benchmark.worker import _NUM_VARIABLES, zdt1


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
    """Verify the sampled true-front hypervolume against closed-form calculus."""

    def test_matches_closed_form_derivation(self) -> None:
        # For reference (1.1, 1.1), integrating the true front
        # f2 = 1 - sqrt(f1) over f1 in [0, 1] gives:
        #   integral_0^1 (1.1 - (1 - sqrt(f1))) df1 + 1.1 * (1.1 - 1)
        #   = (0.1 + 2/3) + 0.11 = 0.87667 (5 s.f.)
        expected = (0.1 + 2 / 3) + 0.11
        result = true_zdt1_hypervolume((1.1, 1.1), samples=20_000)
        self.assertAlmostEqual(result, expected, places=3)


if __name__ == "__main__":
    unittest.main()
