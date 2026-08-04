"""Focused tests for immutable optimizer foundation models."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from black_box_optimizer import (
    AlgorithmSpec,
    CandidateConfiguration,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    ProjectConfiguration,
    StopPolicy,
    WorkerSpec,
)


class FoundationModelTests(unittest.TestCase):
    """Verify basic construction, validation, and immutability."""

    def make_contract(self) -> OptimizationContract:
        return OptimizationContract(
            parameters=(
                ParameterDefinition(
                    name="learning_rate",
                    kind=ParameterKind.FLOAT,
                    minimum=0.0001,
                    maximum=0.1,
                ),
                ParameterDefinition(
                    name="batch_size",
                    kind=ParameterKind.CATEGORICAL,
                    choices=(8, 16, 32),
                ),
            ),
            objectives=(
                Objective(
                    metric_name="validation_accuracy",
                    direction=Direction.MAXIMIZE,
                ),
                Objective(
                    metric_name="validation_loss",
                    direction=Direction.MINIMIZE,
                ),
            ),
        )

    def test_parameter_kinds_and_directions_have_json_values(self) -> None:
        self.assertEqual(ParameterKind.INTEGER.value, "integer")
        self.assertEqual(ParameterKind.FLOAT.value, "float")
        self.assertEqual(ParameterKind.CATEGORICAL.value, "categorical")
        self.assertEqual(Direction.MINIMIZE.value, "minimize")
        self.assertEqual(Direction.MAXIMIZE.value, "maximize")

    def test_numeric_parameter_definitions(self) -> None:
        epochs = ParameterDefinition(
            name="epochs",
            kind=ParameterKind.INTEGER,
            minimum=1,
            maximum=100,
        )
        learning_rate = ParameterDefinition(
            name="learning_rate",
            kind=ParameterKind.FLOAT,
            minimum=0.0001,
            maximum=0.1,
        )

        self.assertEqual(epochs.minimum, 1)
        self.assertEqual(learning_rate.maximum, 0.1)

    def test_categorical_choices_are_copied_to_a_tuple(self) -> None:
        choices = [8, 16, 32]
        definition = ParameterDefinition(
            name="batch_size",
            kind=ParameterKind.CATEGORICAL,
            choices=choices,  # type: ignore[arg-type]
        )
        choices.append(64)

        self.assertEqual(definition.choices, (8, 16, 32))
        with self.assertRaises(FrozenInstanceError):
            definition.name = "changed"  # type: ignore[misc]

    def test_optimization_contract_requires_two_objectives(self) -> None:
        parameter = ParameterDefinition(
            name="epochs",
            kind=ParameterKind.INTEGER,
            minimum=1,
            maximum=10,
        )
        objective = Objective("validation_loss", Direction.MINIMIZE)

        with self.assertRaisesRegex(ValueError, "two objectives"):
            OptimizationContract((parameter,), (objective,))

    def test_worker_command_and_contract_sequences_are_copied(self) -> None:
        command = ["python", "worker.py"]
        worker = WorkerSpec(
            command=command,  # type: ignore[arg-type]
            metrics_argument="--metrics-out",
            timeout_seconds=120.0,
        )
        command.append("unexpected")

        self.assertEqual(worker.command, ("python", "worker.py"))
        self.assertIsInstance(self.make_contract().parameters, tuple)

    def test_complete_project_configuration_is_frozen(self) -> None:
        configuration = ProjectConfiguration(
            worker=WorkerSpec(
                command=("python", "worker.py"),
                metrics_argument="--metrics-out",
                timeout_seconds=120.0,
            ),
            optimization=self.make_contract(),
            algorithm=AlgorithmSpec(name="random_search", seed=42),
            stop_policy=StopPolicy(max_trials=20),
        )

        self.assertEqual(configuration.algorithm.seed, 42)
        with self.assertRaises(FrozenInstanceError):
            configuration.stop_policy = StopPolicy(1)  # type: ignore[misc]

    def test_candidate_copies_and_protects_parameter_mapping(self) -> None:
        source = {"learning_rate": 0.01, "batch_size": 16}
        candidate = CandidateConfiguration(parameters=source)
        source["learning_rate"] = 0.05

        self.assertEqual(candidate.parameters["learning_rate"], 0.01)
        with self.assertRaises(TypeError):
            candidate.parameters["batch_size"] = 32  # type: ignore[index]

    def test_invalid_parameter_domains_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ParameterDefinition(
                name="epochs",
                kind=ParameterKind.INTEGER,
                minimum=10,
                maximum=1,
            )
        with self.assertRaises(ValueError):
            ParameterDefinition(
                name="batch_size",
                kind=ParameterKind.CATEGORICAL,
                choices=(16, 16),
            )

    def test_parameter_definition_rejects_malformed_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"parameter names must match"
        ):
            ParameterDefinition(
                name="1st_epoch",
                kind=ParameterKind.INTEGER,
                minimum=1,
                maximum=10,
            )
        with self.assertRaisesRegex(
            ValueError, r"parameter names must match"
        ):
            ParameterDefinition(
                name="learning rate",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
            )

    def test_parameter_definition_rejects_non_enum_kind(self) -> None:
        with self.assertRaisesRegex(
            TypeError, r"kind must be a ParameterKind"
        ):
            ParameterDefinition(
                name="epochs",
                kind="integer",  # type: ignore[arg-type]
                minimum=1,
                maximum=10,
            )

    def test_integer_domain_rejects_non_integer_bounds(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"integer parameters require integer bounds"
        ):
            ParameterDefinition(
                name="epochs",
                kind=ParameterKind.INTEGER,
                minimum=0.5,
                maximum=10,
            )

    def test_integer_domain_rejects_choices(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"integer parameters cannot declare choices"
        ):
            ParameterDefinition(
                name="epochs",
                kind=ParameterKind.INTEGER,
                minimum=1,
                maximum=10,
                choices=(1, 2),
            )

    def test_float_domain_rejects_non_numeric_bounds(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"float parameters require numeric bounds"
        ):
            ParameterDefinition(
                name="learning_rate",
                kind=ParameterKind.FLOAT,
                minimum="0.0001",  # type: ignore[arg-type]
                maximum=0.1,
            )

    def test_float_domain_rejects_infinite_bounds(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"float parameter bounds must be finite"
        ):
            ParameterDefinition(
                name="learning_rate",
                kind=ParameterKind.FLOAT,
                minimum=math.inf,
                maximum=1.0,
            )

    def test_float_domain_rejects_minimum_at_or_above_maximum(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"float minimum must be less than maximum"
        ):
            ParameterDefinition(
                name="learning_rate",
                kind=ParameterKind.FLOAT,
                minimum=0.5,
                maximum=0.5,
            )

    def test_float_domain_rejects_choices(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"float parameters cannot declare choices"
        ):
            ParameterDefinition(
                name="learning_rate",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                choices=(0.1, 0.2),
            )

    def test_categorical_domain_rejects_bounds(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"categorical parameters cannot declare bounds"
        ):
            ParameterDefinition(
                name="batch_size",
                kind=ParameterKind.CATEGORICAL,
                minimum=0,
                choices=(8, 16, 32),
            )

    def test_categorical_domain_requires_choices(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"categorical parameters require choices"
        ):
            ParameterDefinition(
                name="batch_size",
                kind=ParameterKind.CATEGORICAL,
            )

    def test_categorical_domain_rejects_non_parameter_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"categorical choices must be finite int, float, or str values",
        ):
            ParameterDefinition(
                name="use_dropout",
                kind=ParameterKind.CATEGORICAL,
                choices=(True, False),
            )

    def test_objective_rejects_empty_metric_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"metric_name must be a nonempty string"
        ):
            Objective(metric_name="   ", direction=Direction.MINIMIZE)

    def test_objective_rejects_metric_name_with_surrounding_whitespace(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, r"metric_name cannot have surrounding whitespace"
        ):
            Objective(
                metric_name=" validation_loss",
                direction=Direction.MINIMIZE,
            )

    def test_objective_rejects_non_enum_direction(self) -> None:
        with self.assertRaisesRegex(
            TypeError, r"direction must be a Direction"
        ):
            Objective(
                metric_name="validation_loss",
                direction="minimize",  # type: ignore[arg-type]
            )

    def test_optimization_contract_requires_at_least_one_parameter(
        self,
    ) -> None:
        objectives = (
            Objective("validation_accuracy", Direction.MAXIMIZE),
            Objective("validation_loss", Direction.MINIMIZE),
        )
        with self.assertRaisesRegex(
            ValueError, r"an optimization contract needs a parameter"
        ):
            OptimizationContract((), objectives)

    def test_optimization_contract_rejects_duplicate_parameter_names(
        self,
    ) -> None:
        parameters = (
            ParameterDefinition(
                name="epochs", kind=ParameterKind.INTEGER, minimum=1, maximum=10
            ),
            ParameterDefinition(
                name="epochs", kind=ParameterKind.INTEGER, minimum=1, maximum=20
            ),
        )
        objectives = (
            Objective("validation_accuracy", Direction.MAXIMIZE),
            Objective("validation_loss", Direction.MINIMIZE),
        )
        with self.assertRaisesRegex(
            ValueError, r"parameter names must be unique"
        ):
            OptimizationContract(parameters, objectives)

    def test_optimization_contract_rejects_duplicate_objective_metrics(
        self,
    ) -> None:
        parameters = (
            ParameterDefinition(
                name="epochs", kind=ParameterKind.INTEGER, minimum=1, maximum=10
            ),
        )
        objectives = (
            Objective("validation_loss", Direction.MAXIMIZE),
            Objective("validation_loss", Direction.MINIMIZE),
        )
        with self.assertRaisesRegex(
            ValueError, r"objective metric names must be unique"
        ):
            OptimizationContract(parameters, objectives)

    def test_worker_spec_rejects_empty_command(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"command must contain nonempty strings"
        ):
            WorkerSpec(
                command=(),
                metrics_argument="--metrics-out",
                timeout_seconds=1.0,
            )

    def test_worker_spec_rejects_blank_command_parts(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"command must contain nonempty strings"
        ):
            WorkerSpec(
                command=("python", ""),
                metrics_argument="--metrics-out",
                timeout_seconds=1.0,
            )

    def test_worker_spec_rejects_non_string_metrics_argument(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"metrics_argument must be a string"
        ):
            WorkerSpec(
                command=("python", "worker.py"),
                metrics_argument=123,  # type: ignore[arg-type]
                timeout_seconds=1.0,
            )

    def test_worker_spec_rejects_metrics_argument_without_prefix(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, r"metrics_argument must start with --"
        ):
            WorkerSpec(
                command=("python", "worker.py"),
                metrics_argument="metrics-out",
                timeout_seconds=1.0,
            )

    def test_worker_spec_rejects_non_numeric_timeout(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"timeout_seconds must be numeric"
        ):
            WorkerSpec(
                command=("python", "worker.py"),
                metrics_argument="--metrics-out",
                timeout_seconds="120",  # type: ignore[arg-type]
            )

    def test_worker_spec_rejects_infinite_timeout(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"timeout_seconds must be finite"
        ):
            WorkerSpec(
                command=("python", "worker.py"),
                metrics_argument="--metrics-out",
                timeout_seconds=math.inf,
            )

    def test_worker_spec_rejects_non_positive_timeout(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"timeout_seconds must be positive"
        ):
            WorkerSpec(
                command=("python", "worker.py"),
                metrics_argument="--metrics-out",
                timeout_seconds=0,
            )

    def test_algorithm_spec_rejects_blank_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"algorithm name must be a nonempty string"
        ):
            AlgorithmSpec(name="   ", seed=42)

    def test_algorithm_spec_rejects_non_integer_seed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"seed must be an integer"
        ):
            AlgorithmSpec(
                name="random_search", seed=True  # type: ignore[arg-type]
            )

    def test_stop_policy_rejects_non_integer_max_trials(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"max_trials must be an integer"
        ):
            StopPolicy(max_trials=True)  # type: ignore[arg-type]

    def test_candidate_requires_at_least_one_parameter(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"a candidate needs at least one parameter"
        ):
            CandidateConfiguration(parameters={})

    def test_candidate_rejects_non_finite_parameter_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"candidate values must be finite int, float, or str values",
        ):
            CandidateConfiguration(parameters={"learning_rate": math.nan})


if __name__ == "__main__":
    unittest.main()

