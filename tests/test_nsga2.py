"""Focused tests for NSGA2, covering ranking, crowding, selection, breeding,
generation management, and the failure/exhaustion edge cases documented in
docs/decisions/2026-08-02-nsga2-search-algorithm-design.md.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from black_box_optimizer.models import (
    CandidateConfiguration,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.search.nsga2 import (
    _MAX_DUPLICATE_ATTEMPTS,
    NSGA2,
    _RankedIndividual,
    _breed_one_child,
    _crossover,
    _crowding_distances,
    _default_mutation_rate,
    _default_population_size,
    _mutate,
    _rank_population,
    _sample_new_random_candidate,
    _tournament_select,
)


def make_parameters(count: int) -> tuple[ParameterDefinition, ...]:
    """Build `count` distinct integer parameters, for formula-only tests."""
    return tuple(
        ParameterDefinition(f"p{i}", ParameterKind.INTEGER, 0, 100)
        for i in range(count)
    )


def make_contract(
    *parameters: ParameterDefinition,
    objectives: tuple[Objective, ...] | None = None,
) -> OptimizationContract:
    """Build a contract with the given parameters and two dummy objectives."""
    return OptimizationContract(
        parameters=parameters,
        objectives=objectives
        or (
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


def make_trial_record(
    trial_id: int,
    parameters: dict,
    metrics: dict | None = None,
    execution_status: str = "completed",
    metrics_status: str = "valid",
) -> TrialRecord:
    """Build a TrialRecord, eligible by default, overridable for failures."""
    return TrialRecord(
        trial_id=trial_id,
        parameters=parameters,
        metrics={} if metrics is None else metrics,
        execution_status=execution_status,
        metrics_status=metrics_status,
        runtime_seconds=1.0,
        exit_code=0 if execution_status == "completed" else 1,
        timed_out=execution_status == "timed_out",
    )


def make_ranked(
    trial_id: int, rank: int, crowding_distance: float
) -> _RankedIndividual:
    """Build a _RankedIndividual around a throwaway TrialRecord."""
    record = make_trial_record(
        trial_id, {"x": trial_id}, metrics={"accuracy": 0.5, "loss": 0.5}
    )
    return _RankedIndividual(
        record=record, rank=rank, crowding_distance=crowding_distance
    )


# ---------------------------------------------------------------------------
# Seed validation -- identical contract to RandomSearch's own checks.
# ---------------------------------------------------------------------------


class NSGA2SeedTests(unittest.TestCase):
    """Verify seed validation matches RandomSearch's own rules."""

    def test_boolean_seed_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            NSGA2(seed=True)

    def test_non_integer_seed_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            NSGA2(seed="42")

    def test_negative_seed_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            NSGA2(seed=-5)


# ---------------------------------------------------------------------------
# Default population size / mutation rate -- decisions the original spec
# never made, documented as the file's own choices.
# ---------------------------------------------------------------------------


class DefaultPopulationSizeTests(unittest.TestCase):
    """Verify population_size = clamp(2 * num_parameters, 4, 10)."""

    def test_clamped_up_to_the_minimum_for_few_parameters(self) -> None:
        contract = make_contract(*make_parameters(1))
        self.assertEqual(_default_population_size(contract, None), 4)

    def test_uses_double_parameter_count_in_the_middle_of_the_range(
        self,
    ) -> None:
        contract = make_contract(*make_parameters(3))
        self.assertEqual(_default_population_size(contract, None), 6)

    def test_clamped_down_to_the_maximum_for_many_parameters(self) -> None:
        contract = make_contract(*make_parameters(6))
        self.assertEqual(_default_population_size(contract, None), 10)

    def test_finite_size_clamps_population_below_the_normal_minimum(
        self,
    ) -> None:
        # The bug found via testing: one categorical parameter with only 2
        # legal values can't support the default minimum population of 4.
        contract = make_contract(*make_parameters(1))
        self.assertEqual(_default_population_size(contract, finite_size=2), 2)

    def test_large_finite_size_does_not_force_population_upward(self) -> None:
        contract = make_contract(*make_parameters(1))
        self.assertEqual(
            _default_population_size(contract, finite_size=10_000), 4
        )


class DefaultMutationRateTests(unittest.TestCase):
    """Verify mutation_rate = 1 / num_parameters."""

    def test_targets_one_mutated_parameter_per_child_on_average(self) -> None:
        contract = make_contract(*make_parameters(4))
        self.assertAlmostEqual(_default_mutation_rate(contract), 0.25)

    def test_single_parameter_contract_always_mutates(self) -> None:
        contract = make_contract(*make_parameters(1))
        self.assertAlmostEqual(_default_mutation_rate(contract), 1.0)


# ---------------------------------------------------------------------------
# _rank_population -- non-dominated sorting, reusing pareto.py, plus the
# shared-worst-rank treatment of failed/ineligible trials.
# ---------------------------------------------------------------------------


class RankPopulationTests(unittest.TestCase):
    """Hand-worked fixture: 3 real tiers plus 2 ineligible trials."""

    def _build_generation(self) -> list[TrialRecord]:
        # Tier 0 (mutually non-dominated, each dominates everything below):
        r0 = make_trial_record(0, {}, metrics={"accuracy": 0.9, "loss": 0.2})
        r1 = make_trial_record(1, {}, metrics={"accuracy": 0.8, "loss": 0.1})
        r2 = make_trial_record(2, {}, metrics={"accuracy": 0.85, "loss": 0.15})
        # Tier 1 (dominated by tier 0, mutually non-dominated, dominate r5):
        r3 = make_trial_record(3, {}, metrics={"accuracy": 0.7, "loss": 0.3})
        r4 = make_trial_record(4, {}, metrics={"accuracy": 0.65, "loss": 0.25})
        # Tier 2 (dominated by everything above):
        r5 = make_trial_record(5, {}, metrics={"accuracy": 0.5, "loss": 0.4})
        # Ineligible, in two different ways:
        f0 = make_trial_record(
            6, {}, execution_status="process_failed", metrics_status="missing"
        )
        f1 = make_trial_record(7, {}, metrics_status="nonfinite")
        return [r0, r1, r2, r3, r4, r5, f0, f1]

    def test_hand_worked_ranks_match_expected_tiers(self) -> None:
        contract = make_contract(*make_parameters(1))
        generation = self._build_generation()

        ranked = _rank_population(generation, contract)

        ranks_by_id = {
            individual.record.trial_id: individual.rank
            for individual in ranked
        }
        self.assertEqual(
            ranks_by_id,
            {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 3},
        )

    def test_ineligible_trials_get_zero_crowding_distance(self) -> None:
        contract = make_contract(*make_parameters(1))
        generation = self._build_generation()

        ranked = _rank_population(generation, contract)

        crowding_by_id = {
            individual.record.trial_id: individual.crowding_distance
            for individual in ranked
        }
        self.assertEqual(crowding_by_id[6], 0.0)
        self.assertEqual(crowding_by_id[7], 0.0)

    def test_every_record_gets_exactly_one_rank(self) -> None:
        contract = make_contract(*make_parameters(1))
        generation = self._build_generation()

        ranked = _rank_population(generation, contract)

        self.assertEqual(len(ranked), len(generation))

    def test_all_failed_generation_shares_rank_zero(self) -> None:
        # No eligible records means no real tiers exist, so "one worse than
        # the last real tier" is rank 0 -- there's nothing successful for
        # them to lose to.
        contract = make_contract(*make_parameters(1))
        generation = [
            make_trial_record(
                0, {}, execution_status="launch_failed", metrics_status="missing"
            ),
            make_trial_record(
                1, {}, execution_status="timed_out", metrics_status="missing"
            ),
        ]

        ranked = _rank_population(generation, contract)

        self.assertTrue(all(individual.rank == 0 for individual in ranked))

    def test_does_not_mutate_input_records(self) -> None:
        contract = make_contract(*make_parameters(1))
        generation = self._build_generation()
        before = [dict(record.metrics) for record in generation]

        _rank_population(generation, contract)

        after = [dict(record.metrics) for record in generation]
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# _crowding_distances
# ---------------------------------------------------------------------------


class CrowdingDistanceTests(unittest.TestCase):
    def test_boundary_and_interior_points_in_a_three_way_tier(self) -> None:
        low = make_trial_record(0, {}, metrics={"accuracy": 0.0, "loss": 1.0})
        mid = make_trial_record(1, {}, metrics={"accuracy": 0.4, "loss": 0.6})
        high = make_trial_record(2, {}, metrics={"accuracy": 1.0, "loss": 0.0})
        contract = make_contract(*make_parameters(1))

        distances = _crowding_distances((low, mid, high), contract)

        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[2], float("inf"))
        self.assertEqual(distances[1], 2.0)

    def test_tier_of_one_is_entirely_boundary(self) -> None:
        only = make_trial_record(0, {}, metrics={"accuracy": 0.5, "loss": 0.5})
        contract = make_contract(*make_parameters(1))

        distances = _crowding_distances((only,), contract)

        self.assertEqual(distances[0], float("inf"))

    def test_tier_of_two_is_entirely_boundary(self) -> None:
        a = make_trial_record(0, {}, metrics={"accuracy": 0.5, "loss": 0.5})
        b = make_trial_record(1, {}, metrics={"accuracy": 0.9, "loss": 0.1})
        contract = make_contract(*make_parameters(1))

        distances = _crowding_distances((a, b), contract)

        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[1], float("inf"))

    def test_zero_range_objective_is_skipped_without_dividing_by_zero(
        self,
    ) -> None:
        # Accuracy is tied across all three -- that objective must not
        # crash and must not contribute to the interior point's distance.
        a = make_trial_record(0, {}, metrics={"accuracy": 0.5, "loss": 1.0})
        b = make_trial_record(1, {}, metrics={"accuracy": 0.5, "loss": 0.6})
        c = make_trial_record(2, {}, metrics={"accuracy": 0.5, "loss": 0.0})
        contract = make_contract(*make_parameters(1))

        distances = _crowding_distances((a, b, c), contract)

        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[2], float("inf"))
        self.assertEqual(distances[1], 1.0)

    def test_tied_objective_never_marks_a_boundary_regardless_of_input_order(
        self,
    ) -> None:
        # Real bug, since fixed: boundary marking used to happen before
        # the tied-objective check, so a tied objective would mark
        # whichever records landed first/last in tier_records -- an
        # accident of input order, not a real extreme -- as infinite.
        # Feeding the same four records in two different orders must
        # produce identical distances; only loss (not tied) should ever
        # contribute a boundary or a gap here.
        p = make_trial_record(0, {}, metrics={"accuracy": 0.5, "loss": 0.1})
        q = make_trial_record(1, {}, metrics={"accuracy": 0.5, "loss": 0.4})
        r = make_trial_record(2, {}, metrics={"accuracy": 0.5, "loss": 0.6})
        s = make_trial_record(3, {}, metrics={"accuracy": 0.5, "loss": 0.9})
        contract = make_contract(*make_parameters(1))

        first_order = _crowding_distances((r, p, s, q), contract)
        second_order = _crowding_distances((p, q, r, s), contract)

        expected = {0: float("inf"), 1: 0.625, 2: 0.625, 3: float("inf")}
        self.assertEqual(first_order, expected)
        self.assertEqual(second_order, expected)

    def test_a_boundary_point_on_one_objective_short_circuits_on_the_next(
        self,
    ) -> None:
        # A is accuracy's max (a boundary -> inf) but sits in loss's
        # interior position. By the time the loss objective is processed,
        # A already carries inf from accuracy -- the interior loop must
        # recognize that and skip adding a finite loss-based gap on top of
        # it, rather than only ever seeing "already inf" at true
        # boundary indices.
        a = make_trial_record(0, {}, metrics={"accuracy": 0.9, "loss": 0.5})
        b = make_trial_record(1, {}, metrics={"accuracy": 0.5, "loss": 0.9})
        c = make_trial_record(2, {}, metrics={"accuracy": 0.1, "loss": 0.1})
        contract = make_contract(*make_parameters(1))

        distances = _crowding_distances((a, b, c), contract)

        # C is a boundary on both objectives, B is loss's max, and A is
        # accuracy's max -- every record ends up a boundary on at least
        # one objective in this fixture, all landing on infinity.
        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[1], float("inf"))
        self.assertEqual(distances[2], float("inf"))


# ---------------------------------------------------------------------------
# _tournament_select
# ---------------------------------------------------------------------------


class TournamentSelectTests(unittest.TestCase):
    def test_lower_rank_wins_regardless_of_crowding_distance(self) -> None:
        better = make_ranked(trial_id=0, rank=0, crowding_distance=0.0)
        worse = make_ranked(trial_id=1, rank=1, crowding_distance=999.0)
        population = [better, worse]
        generator = Mock()
        generator.choice.return_value = np.array([0, 1])

        winner = _tournament_select(population, generator)

        self.assertEqual(winner.record.trial_id, 0)

    def test_higher_crowding_distance_wins_a_rank_tie(self) -> None:
        cramped = make_ranked(trial_id=0, rank=0, crowding_distance=0.1)
        spread_out = make_ranked(trial_id=1, rank=0, crowding_distance=5.0)
        population = [cramped, spread_out]
        generator = Mock()
        generator.choice.return_value = np.array([0, 1])

        winner = _tournament_select(population, generator)

        self.assertEqual(winner.record.trial_id, 1)

    def test_genuine_tie_returns_the_first_drawn_individual(self) -> None:
        first = make_ranked(trial_id=0, rank=0, crowding_distance=1.0)
        second = make_ranked(trial_id=1, rank=0, crowding_distance=1.0)
        population = [first, second]
        generator = Mock()
        generator.choice.return_value = np.array([0, 1])

        winner = _tournament_select(population, generator)

        self.assertEqual(winner.record.trial_id, 0)


# ---------------------------------------------------------------------------
# _crossover
# ---------------------------------------------------------------------------


class CrossoverTests(unittest.TestCase):
    def test_each_parameter_comes_from_exactly_one_parent(self) -> None:
        contract = make_contract(
            ParameterDefinition("a", ParameterKind.INTEGER, 0, 100),
            ParameterDefinition("b", ParameterKind.INTEGER, 0, 100),
        )
        parent_a = make_trial_record(0, {"a": 1, "b": 2})
        parent_b = make_trial_record(1, {"a": 11, "b": 22})
        generator = Mock()
        # First draw < 0.5 -> parent_a's value; second draw >= 0.5 -> parent_b's.
        generator.random.side_effect = [0.1, 0.9]

        child = _crossover(parent_a, parent_b, contract, generator)

        self.assertEqual(child, {"a": 1, "b": 22})

    def test_never_invents_a_blended_value(self) -> None:
        contract = make_contract(
            ParameterDefinition("a", ParameterKind.INTEGER, 0, 100)
        )
        parent_a = make_trial_record(0, {"a": 10})
        parent_b = make_trial_record(1, {"a": 90})
        generator = np.random.default_rng(0)

        for _ in range(20):
            child = _crossover(parent_a, parent_b, contract, generator)
            self.assertIn(child["a"], (10, 90))


# ---------------------------------------------------------------------------
# _mutate
# ---------------------------------------------------------------------------


class MutateTests(unittest.TestCase):
    def test_mutation_rate_zero_never_changes_anything(self) -> None:
        contract = make_contract(*make_parameters(3))
        parameters = {"p0": 1, "p1": 2, "p2": 3}
        generator = np.random.default_rng(0)

        mutated = _mutate(parameters, contract, generator, mutation_rate=0.0)

        self.assertEqual(mutated, parameters)

    def test_mutation_rate_one_replaces_every_parameter(self) -> None:
        contract = make_contract(*make_parameters(3))
        parameters = {"p0": 1, "p1": 2, "p2": 3}
        generator = np.random.default_rng(0)

        with patch(
            "black_box_optimizer.search.nsga2._sample_value",
            return_value=999,
        ):
            mutated = _mutate(
                parameters, contract, generator, mutation_rate=1.0
            )

        self.assertEqual(mutated, {"p0": 999, "p1": 999, "p2": 999})

    def test_does_not_mutate_the_input_dict_in_place(self) -> None:
        contract = make_contract(*make_parameters(1))
        parameters = {"p0": 1}
        generator = np.random.default_rng(0)

        _mutate(parameters, contract, generator, mutation_rate=1.0)

        self.assertEqual(parameters, {"p0": 1})


# ---------------------------------------------------------------------------
# Duplicate-avoidance retry limits, mirroring RandomSearch's own bound.
# ---------------------------------------------------------------------------


class DuplicateHandlingTests(unittest.TestCase):
    def test_sample_new_random_candidate_gives_up_after_max_attempts(
        self,
    ) -> None:
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1)
        )
        generator = np.random.default_rng(0)
        forbidden = {(("x", 0),)}

        with patch(
            "black_box_optimizer.search.nsga2._sample_value",
            return_value=0,
        ):
            result = _sample_new_random_candidate(
                contract, generator, forbidden
            )

        self.assertIsNone(result)

    def test_breed_one_child_gives_up_after_max_attempts(self) -> None:
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1)
        )
        generator = np.random.default_rng(0)
        population = [
            make_ranked(trial_id=0, rank=0, crowding_distance=1.0),
            make_ranked(trial_id=1, rank=1, crowding_distance=1.0),
        ]
        forbidden = {(("x", 7),)}

        with (
            patch(
                "black_box_optimizer.search.nsga2._crossover",
                return_value={"x": 7},
            ),
            patch(
                "black_box_optimizer.search.nsga2._mutate",
                side_effect=lambda parameters, *_args, **_kwargs: dict(
                    parameters
                ),
            ),
        ):
            result = _breed_one_child(
                population, contract, generator, 0.5, forbidden
            )

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# NSGA2.propose() -- the queue mechanism, generation boundaries, and
# multi-generation integration behavior.
# ---------------------------------------------------------------------------


class NSGA2GenerationZeroTests(unittest.TestCase):
    def test_first_generation_is_random_and_fully_unique(self) -> None:
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000)
        )
        algorithm = NSGA2(seed=3)
        history: list[TrialRecord] = []
        seen = set()

        for trial_id in range(4):  # 1 parameter -> population_size 4
            result = algorithm.propose(contract, history)
            self.assertEqual(result.status, "candidate")
            seen.add(result.candidate.parameters["x"])
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

        self.assertEqual(len(seen), 4)

    def test_same_seed_and_history_reproduce_the_first_proposal(self) -> None:
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000)
        )

        first = NSGA2(seed=7).propose(contract, [])
        second = NSGA2(seed=7).propose(contract, [])

        self.assertEqual(
            first.candidate.parameters, second.candidate.parameters
        )


class NSGA2QueueAndBreedingTests(unittest.TestCase):
    """Verify the pending_children queue and generation-boundary breeding."""

    def test_children_only_use_parameter_values_seen_in_the_parent_generation(
        self,
    ) -> None:
        # Two parameters, so uniform crossover has a real combinatorial
        # space to draw from (up to 4x4 pairings) rather than being able to
        # produce at most population_size distinct outcomes -- with
        # mutation stubbed to a no-op, every bred child's per-parameter
        # value must still come from generation 0.
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000),
            ParameterDefinition("y", ParameterKind.INTEGER, 0, 1000),
        )
        algorithm = NSGA2(seed=11)
        history: list[TrialRecord] = []
        generation_zero_x = set()
        generation_zero_y = set()

        for trial_id in range(4):
            result = algorithm.propose(contract, history)
            generation_zero_x.add(result.candidate.parameters["x"])
            generation_zero_y.add(result.candidate.parameters["y"])
            history.append(
                make_trial_record(
                    trial_id,
                    result.candidate.parameters,
                    metrics={
                        "accuracy": trial_id / 10,
                        "loss": 1 - trial_id / 10,
                    },
                )
            )

        with patch(
            "black_box_optimizer.search.nsga2._mutate",
            side_effect=lambda parameters, *_args, **_kwargs: dict(
                parameters
            ),
        ):
            for _ in range(4):
                result = algorithm.propose(contract, history)
                self.assertEqual(result.status, "candidate")
                self.assertIn(
                    result.candidate.parameters["x"], generation_zero_x
                )
                self.assertIn(
                    result.candidate.parameters["y"], generation_zero_y
                )

    def test_seeded_multi_generation_run_is_fully_reproducible(self) -> None:
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000)
        )

        def run(seed: int) -> list[int]:
            algorithm = NSGA2(seed=seed)
            history: list[TrialRecord] = []
            proposed = []
            for trial_id in range(12):  # 3 generations of 4
                result = algorithm.propose(contract, history)
                x = result.candidate.parameters["x"]
                proposed.append(x)
                history.append(
                    make_trial_record(
                        trial_id,
                        {"x": x},
                        metrics={"accuracy": x, "loss": -x},
                    )
                )
            return proposed

        self.assertEqual(run(seed=99), run(seed=99))


class NSGA2FailureHandlingIntegrationTests(unittest.TestCase):
    def test_a_generation_with_some_failed_trials_still_breeds_cleanly(
        self,
    ) -> None:
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000)
        )
        algorithm = NSGA2(seed=13)
        history: list[TrialRecord] = []

        for trial_id in range(4):
            result = algorithm.propose(contract, history)
            if trial_id % 2 == 0:
                record = make_trial_record(
                    trial_id,
                    result.candidate.parameters,
                    metrics={"accuracy": 0.9, "loss": 0.1},
                )
            else:
                record = make_trial_record(
                    trial_id,
                    result.candidate.parameters,
                    execution_status="process_failed",
                    metrics_status="missing",
                )
            history.append(record)

        # Breeding generation 1 must not crash on the failed trials, and
        # must still produce a real candidate.
        result = algorithm.propose(contract, history)
        self.assertEqual(result.status, "candidate")

    def test_an_entirely_failed_generation_still_breeds(self) -> None:
        # Every trial ineligible -- _rank_population's "all failed" branch
        # (rank 0 for everyone) must still let tournament selection and
        # crossover run without a real Pareto tier to draw from.
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000)
        )
        algorithm = NSGA2(seed=17)
        history: list[TrialRecord] = []

        for trial_id in range(4):
            result = algorithm.propose(contract, history)
            history.append(
                make_trial_record(
                    trial_id,
                    result.candidate.parameters,
                    execution_status="process_failed",
                    metrics_status="missing",
                )
            )

        result = algorithm.propose(contract, history)
        self.assertEqual(result.status, "candidate")


class NSGA2ExhaustionTests(unittest.TestCase):
    def test_generation_zero_duplicate_stall_returns_proposal_failed(
        self,
    ) -> None:
        # test_near_exhausted_space_currently_reports_proposal_failed below
        # already exercises this exit through _breed_next_generation.
        # _breed_random_generation has the exact same guard for generation
        # 0's own random seeding, reached before any real generation has
        # ever completed -- this forces that path specifically, the same
        # way RandomSearch's own duplicate-stall test does.
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 1000)
        )
        algorithm = NSGA2(seed=1)

        with patch(
            "black_box_optimizer.search.nsga2._sample_value",
            return_value=5,
        ):
            result = algorithm.propose(contract, [])

        self.assertEqual(result.status, "proposal_failed")
        self.assertIsNone(result.candidate)
        self.assertIn("100 attempts per candidate", result.reason)

    def test_population_clamped_to_a_tiny_finite_space_still_proposes(
        self,
    ) -> None:
        # Without the finite-size clamp, the default floor of 4 would be
        # demanded from a space that only contains 2 candidates.
        contract = make_contract(
            ParameterDefinition(
                "mode", ParameterKind.CATEGORICAL, choices=("a", "b")
            )
        )
        algorithm = NSGA2(seed=1)

        result = algorithm.propose(contract, [])

        self.assertEqual(result.status, "candidate")

    def test_tiny_finite_space_reports_search_exhausted_not_proposal_failed(
        self,
    ) -> None:
        contract = make_contract(
            ParameterDefinition(
                "mode", ParameterKind.CATEGORICAL, choices=("a", "b")
            )
        )
        algorithm = NSGA2(seed=1)
        history: list[TrialRecord] = []

        for trial_id in range(2):
            result = algorithm.propose(contract, history)
            self.assertEqual(result.status, "candidate")
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

        result = algorithm.propose(contract, history)
        self.assertEqual(result.status, "search_exhausted")
        self.assertIsNone(result.candidate)

    def test_near_exhausted_space_currently_reports_proposal_failed(
        self,
    ) -> None:
        # Documents a known, open limitation (see "Better finite-space
        # handling" in the design doc): population_size is fixed once at
        # the first propose() call and never shrinks. A space with 6 legal
        # values and a population of 4 leaves only 2 unique candidates for
        # the second generation, which can't fill a batch of 4 -- this
        # currently surfaces as proposal_failed, not a shrunk generation or
        # search_exhausted. If that gets fixed, this test's expected status
        # should change too.
        contract = make_contract(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 5)
        )
        algorithm = NSGA2(seed=1)
        history: list[TrialRecord] = []

        for trial_id in range(4):
            result = algorithm.propose(contract, history)
            self.assertEqual(result.status, "candidate")
            history.append(
                make_trial_record(
                    trial_id,
                    result.candidate.parameters,
                    metrics={"accuracy": 0.5, "loss": 0.5},
                )
            )

        result = algorithm.propose(contract, history)
        self.assertEqual(result.status, "proposal_failed")
        self.assertIn("100 attempts per candidate", result.reason)


if __name__ == "__main__":
    unittest.main()
