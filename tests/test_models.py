"""Focused tests for immutable optimizer foundation models."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

