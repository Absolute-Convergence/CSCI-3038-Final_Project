"""Focused tests for RandomSearch, per TDS sections 7.2 through 7.5."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from black_box_optimizer.models import (
    CandidateConfiguration,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.search.random_search import RandomSearch, candidate_key


def make_contract(*parameters: ParameterDefinition) -> OptimizationContract:
    """Build a contract with the given parameters and two dummy objectives."""
    return OptimizationContract(
        parameters=parameters,
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


def make_trial_record(trial_id: int, parameters: dict) -> TrialRecord:
    """Build a minimal, valid TrialRecord carrying the given parameters."""
    return TrialRecord(
        trial_id=trial_id,
        parameters=parameters,
        metrics={"accuracy": 0.9, "loss": 0.1},
        execution_status="completed",
        metrics_status="valid",
        runtime_seconds=1.0,
        exit_code=0,
        timed_out=False,
    )


class CandidateKeyTests(unittest.TestCase):
    """Verify the canonical candidate key follows contract order."""

    def test_key_follows_contract_order_not_dict_order(self) -> None:
        contract = make_contract(
            ParameterDefinition("b", ParameterKind.INTEGER, 0, 10),
            ParameterDefinition("a", ParameterKind.INTEGER, 0, 10),
        )
        # Insertion order is a-then-b, the opposite of contract order.
        candidate = CandidateConfiguration(parameters={"a": 1, "b": 2})

        key = candidate_key(contract, candidate)

        self.assertEqual(key, (("b", 2), ("a", 1)))

    def test_extra_candidate_keys_are_ignored(self) -> None:
        # candidate_key() iterates contract.parameters, not the
        # candidate's own keys, so anything extra just gets skipped.
        contract = make_contract(
            ParameterDefinition("a", ParameterKind.INTEGER, 0, 10)
        )
        candidate = CandidateConfiguration(parameters={"a": 1, "extra": 99})

        key = candidate_key(contract, candidate)

        self.assertEqual(key, (("a", 1),))


class RandomSearchSeedTests(unittest.TestCase):
    """Verify seed validation and the reproducibility guarantee."""

    def test_boolean_seed_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            RandomSearch(seed=True)

    def test_non_integer_seed_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            RandomSearch(seed="42")

    def test_negative_seed_rejected(self) -> None:
        # Without our own check, this fails with a raw NumPy internal
        # message instead of a message consistent with the rest of the
        # project's error handling.
        with self.assertRaisesRegex(ValueError, "negative"):
            RandomSearch(seed=-5)

    def test_same_seed_and_history_produce_same_proposal(self) -> None:
        contract = make_contract(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.0001, 0.1
            ),
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8, 16, 32)
            ),
        )

        first = RandomSearch(seed=7).propose(contract, [])
        second = RandomSearch(seed=7).propose(contract, [])

        self.assertEqual(
            first.candidate.parameters, second.candidate.parameters
        )

    def test_replaying_a_full_run_reproduces_every_proposal(self) -> None:
        # The reproducibility guarantee is about replaying an entire run
        # from the same seed with a matching growing history, not about
        # calling propose() twice on one already-advanced instance with
        # an unchanged history -- a real run never does the latter, since
        # history always grows between calls.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 100)
        )

        def run(seed: int) -> list[int]:
            algorithm = RandomSearch(seed=seed)
            history: list[TrialRecord] = []
            proposed = []
            for trial_id in range(3):
                result = algorithm.propose(contract, history)
                proposed.append(result.candidate.parameters["epochs"])
                history.append(
                    make_trial_record(trial_id, result.candidate.parameters)
                )
            return proposed

        self.assertEqual(run(seed=99), run(seed=99))

    def test_same_seed_and_nonempty_history_produce_same_proposal(self) -> None:
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 100)
        )
        history = [make_trial_record(0, {"epochs": 47})]

        first = RandomSearch(seed=7).propose(contract, history)
        second = RandomSearch(seed=7).propose(contract, history)

        self.assertEqual(first.status, second.status)
        self.assertEqual(
            first.candidate.parameters, second.candidate.parameters
        )


class RandomSearchSamplingTests(unittest.TestCase):
    """Verify sampled values respect each parameter kind's declared domain."""

    def test_integer_values_stay_within_bounds(self) -> None:
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 5, 8)
        )
        algorithm = RandomSearch(seed=1)
        history: list[TrialRecord] = []

        for trial_id in range(20):
            result = algorithm.propose(contract, history)
            if result.status != "candidate":
                break
            value = result.candidate.parameters["epochs"]
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 5)
            self.assertLessEqual(value, 8)
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

    def test_both_integer_endpoints_are_actually_sampled(self) -> None:
        # Exhausting a tiny domain proves the sampler can actually reach
        # both bounds through real sampling, not just via constructed
        # history -- an off-by-one in the +1 inclusive-bound math would
        # either miss the maximum or make exhaustion never trigger.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 3)
        )
        algorithm = RandomSearch(seed=1)
        history: list[TrialRecord] = []
        seen = set()

        for trial_id in range(50):
            result = algorithm.propose(contract, history)
            if result.status != "candidate":
                break
            seen.add(result.candidate.parameters["epochs"])
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

        self.assertEqual(seen, {0, 1, 2, 3})

    def test_float_values_stay_within_bounds(self) -> None:
        contract = make_contract(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.0001, 0.1
            )
        )
        algorithm = RandomSearch(seed=1)

        result = algorithm.propose(contract, [])
        self.assertEqual(result.status, "candidate")
        self.assertIsNotNone(result.candidate)
        value = result.candidate.parameters["learning_rate"]
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, 0.0001)
        self.assertLess(value, 0.1)

    def test_categorical_values_come_from_declared_choices(self) -> None:
        contract = make_contract(
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8, 16, 32)
            )
        )
        algorithm = RandomSearch(seed=1)
        history: list[TrialRecord] = []

        for trial_id in range(10):
            result = algorithm.propose(contract, history)
            if result.status != "candidate":
                break
            value = result.candidate.parameters["batch_size"]
            self.assertIn(value, (8, 16, 32))
            self.assertIsInstance(value, int)
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

    def test_mixed_type_choices_keep_their_original_type(self) -> None:
        # Without dtype=object, NumPy would silently convert the integer 1
        # into the string "1" since it has to pick one common array type.
        contract = make_contract(
            ParameterDefinition(
                "mode", ParameterKind.CATEGORICAL, choices=(1, "small")
            )
        )
        algorithm = RandomSearch(seed=1)
        history: list[TrialRecord] = []
        seen_types = set()

        for trial_id in range(20):
            result = algorithm.propose(contract, history)
            if result.status != "candidate":
                break
            value = result.candidate.parameters["mode"]
            self.assertIn(value, (1, "small"))
            if value == 1:
                self.assertIsInstance(value, int)
            else:
                self.assertIsInstance(value, str)
            seen_types.add(type(value))
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

        # Confirms both choices were actually sampled during this run,
        # not just the one that happens to survive type coercion.
        self.assertEqual(seen_types, {int, str})


class RandomSearchExhaustionTests(unittest.TestCase):
    """Verify duplicate avoidance and finite-space exhaustion."""

    def test_avoids_repeating_a_candidate_already_in_history(self) -> None:
        # Only two legal values exist: 0 and 1.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 1)
        )
        history = [make_trial_record(0, {"epochs": 0})]
        algorithm = RandomSearch(seed=1)

        result = algorithm.propose(contract, history)

        self.assertEqual(result.status, "candidate")
        self.assertEqual(result.candidate.parameters["epochs"], 1)

    def test_exhausted_finite_space_returns_search_exhausted(self) -> None:
        # Only two legal values exist, and both are already attempted.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 1)
        )
        history = [
            make_trial_record(0, {"epochs": 0}),
            make_trial_record(1, {"epochs": 1}),
        ]
        algorithm = RandomSearch(seed=1)

        result = algorithm.propose(contract, history)

        self.assertEqual(result.status, "search_exhausted")
        self.assertIsNone(result.candidate)

    def test_failed_attempts_still_count_toward_exhaustion(self) -> None:
        # A failed trial still occupies its candidate slot and must not be
        # silently resampled.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 1)
        )
        failed_record = TrialRecord(
            trial_id=0,
            parameters={"epochs": 0},
            metrics={},
            execution_status="process_failed",
            metrics_status="missing",
            runtime_seconds=0.5,
            exit_code=1,
            timed_out=False,
        )
        history = [failed_record, make_trial_record(1, {"epochs": 1})]
        algorithm = RandomSearch(seed=1)

        result = algorithm.propose(contract, history)

        self.assertEqual(result.status, "search_exhausted")

    def test_duplicate_history_records_do_not_fake_exhaustion(self) -> None:
        # Exhaustion must compare distinct attempted keys, not raw history
        # length -- two records with the same parameters should count once.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 1)
        )
        history = [
            make_trial_record(0, {"epochs": 0}),
            make_trial_record(1, {"epochs": 0}),
        ]
        algorithm = RandomSearch(seed=1)

        result = algorithm.propose(contract, history)

        self.assertEqual(result.status, "candidate")
        self.assertEqual(result.candidate.parameters["epochs"], 1)

    def test_one_value_domain_exhausts_after_one_attempt(self) -> None:
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 5, 5)
        )
        algorithm = RandomSearch(seed=1)

        first = algorithm.propose(contract, [])
        self.assertEqual(first.status, "candidate")
        self.assertEqual(first.candidate.parameters["epochs"], 5)

        history = [make_trial_record(0, {"epochs": 5})]
        second = algorithm.propose(contract, history)
        self.assertEqual(second.status, "search_exhausted")

    def test_mixed_integer_and_categorical_exhaustion(self) -> None:
        # 2 integer values x 2 categorical choices = 4 total candidates.
        contract = make_contract(
            ParameterDefinition("epochs", ParameterKind.INTEGER, 0, 1),
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8, 16)
            ),
        )
        algorithm = RandomSearch(seed=1)
        history: list[TrialRecord] = []

        for trial_id in range(4):
            result = algorithm.propose(contract, history)
            self.assertEqual(result.status, "candidate")
            history.append(
                make_trial_record(trial_id, result.candidate.parameters)
            )

        exhausted = algorithm.propose(contract, history)
        self.assertEqual(exhausted.status, "search_exhausted")

    def test_float_parameter_space_is_never_treated_as_exhausted(self) -> None:
        # A space containing a float parameter has no finite cardinality.
        contract = make_contract(
            ParameterDefinition("learning_rate", ParameterKind.FLOAT, 0.0, 1.0)
        )
        algorithm = RandomSearch(seed=1)

        result = algorithm.propose(contract, [])

        self.assertEqual(result.status, "candidate")

    def test_duplicate_sampling_stall_returns_proposal_failed(self) -> None:
        # A float space can never report search_exhausted, so this is the
        # only way to reach proposal_failed: force every sample to repeat
        # an already-attempted value until the retry limit runs out.
        contract = make_contract(
            ParameterDefinition("learning_rate", ParameterKind.FLOAT, 0.0, 1.0)
        )
        history = [make_trial_record(0, {"learning_rate": 0.5})]
        algorithm = RandomSearch(seed=1)

        with patch(
            "black_box_optimizer.search.random_search._sample_value",
            return_value=0.5,
        ):
            result = algorithm.propose(contract, history)

        self.assertEqual(result.status, "proposal_failed")
        self.assertIsNone(result.candidate)
        self.assertIsNotNone(result.reason)
        self.assertIn("100 attempts", result.reason)


if __name__ == "__main__":
    unittest.main()
