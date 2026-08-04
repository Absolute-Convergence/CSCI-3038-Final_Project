"""Focused tests for declared-domain candidate validation."""

from __future__ import annotations

import unittest

from black_box_optimizer.candidate_validation import (
    CandidateValidationError,
    validate_candidate,
)
from black_box_optimizer.models import (
    CandidateConfiguration,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)


def make_contract() -> OptimizationContract:
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "epochs", ParameterKind.INTEGER, minimum=1, maximum=10
            ),
            ParameterDefinition(
                "learning_rate",
                ParameterKind.FLOAT,
                minimum=0.001,
                maximum=0.1,
            ),
            ParameterDefinition(
                "mode",
                ParameterKind.CATEGORICAL,
                choices=("fast", "accurate"),
            ),
        ),
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


class CandidateValidationTests(unittest.TestCase):
    def test_valid_candidate_is_returned_in_contract_order(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "mode": "fast",
                "epochs": 4,
                "learning_rate": 0.01,
            }
        )

        validated = validate_candidate(candidate, make_contract())

        self.assertEqual(
            tuple(validated.parameters),
            ("epochs", "learning_rate", "mode"),
        )
        self.assertEqual(dict(validated.parameters), dict(candidate.parameters))

    def test_numeric_bounds_are_inclusive(self) -> None:
        contract = make_contract()
        low = CandidateConfiguration(
            parameters={
                "epochs": 1,
                "learning_rate": 0.001,
                "mode": "fast",
            }
        )
        high = CandidateConfiguration(
            parameters={
                "epochs": 10,
                "learning_rate": 0.1,
                "mode": "accurate",
            }
        )

        validate_candidate(low, contract)
        validate_candidate(high, contract)

    def test_integer_parameter_rejects_float(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "epochs": 4.0,
                "learning_rate": 0.01,
                "mode": "fast",
            }
        )

        with self.assertRaisesRegex(CandidateValidationError, "epochs"):
            validate_candidate(candidate, make_contract())

    def test_float_parameter_accepts_integer_in_range(self) -> None:
        contract = OptimizationContract(
            parameters=(
                ParameterDefinition(
                    "scale", ParameterKind.FLOAT, minimum=0.0, maximum=2.0
                ),
            ),
            objectives=(
                Objective("accuracy", Direction.MAXIMIZE),
                Objective("loss", Direction.MINIMIZE),
            ),
        )
        candidate = CandidateConfiguration(parameters={"scale": 1})

        validated = validate_candidate(candidate, contract)

        self.assertEqual(validated.parameters["scale"], 1)

    def test_out_of_range_numeric_value_is_rejected(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "epochs": 11,
                "learning_rate": 0.01,
                "mode": "fast",
            }
        )

        with self.assertRaisesRegex(CandidateValidationError, "through 10"):
            validate_candidate(candidate, make_contract())

    def test_unknown_categorical_value_is_rejected(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "epochs": 4,
                "learning_rate": 0.01,
                "mode": "turbo",
            }
        )

        with self.assertRaisesRegex(CandidateValidationError, "mode"):
            validate_candidate(candidate, make_contract())

    def test_categorical_integer_choice_rejects_equal_float(self) -> None:
        contract = OptimizationContract(
            parameters=(
                ParameterDefinition(
                    "batch_size",
                    ParameterKind.CATEGORICAL,
                    choices=(8, 16, 32),
                ),
            ),
            objectives=(
                Objective("accuracy", Direction.MAXIMIZE),
                Objective("loss", Direction.MINIMIZE),
            ),
        )
        candidate = CandidateConfiguration(parameters={"batch_size": 8.0})

        with self.assertRaisesRegex(CandidateValidationError, "batch_size"):
            validate_candidate(candidate, contract)

    def test_float_parameter_rejects_unrepresentably_huge_integer(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "epochs": 4,
                "learning_rate": 10**400,
                "mode": "fast",
            }
        )

        with self.assertRaisesRegex(
            CandidateValidationError, "learning_rate"
        ):
            validate_candidate(candidate, make_contract())

    def test_missing_and_extra_parameters_are_reported_together(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "epochs": 4,
                "learning_rate": 0.01,
                "other": "fast",
            }
        )

        with self.assertRaises(CandidateValidationError) as raised:
            validate_candidate(candidate, make_contract())

        message = str(raised.exception)
        self.assertIn("missing parameters: mode", message)
        self.assertIn("unexpected parameters: other", message)

    def test_missing_parameters_alone_are_reported(self) -> None:
        candidate = CandidateConfiguration(
            parameters={"epochs": 4, "learning_rate": 0.01}
        )

        with self.assertRaises(CandidateValidationError) as raised:
            validate_candidate(candidate, make_contract())

        message = str(raised.exception)
        self.assertIn("missing parameters: mode", message)
        self.assertNotIn("unexpected parameters", message)

    def test_extra_parameters_alone_are_reported(self) -> None:
        candidate = CandidateConfiguration(
            parameters={
                "epochs": 4,
                "learning_rate": 0.01,
                "mode": "fast",
                "bonus": "unexpected",
            }
        )

        with self.assertRaises(CandidateValidationError) as raised:
            validate_candidate(candidate, make_contract())

        message = str(raised.exception)
        self.assertNotIn("missing parameters", message)
        self.assertIn("unexpected parameters: bonus", message)


if __name__ == "__main__":
    unittest.main()
