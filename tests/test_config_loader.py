"""Focused tests for project configuration loading and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from black_box_optimizer import (
    ConfigurationError,
    Direction,
    ParameterKind,
    load_configuration,
)


class ConfigurationLoaderTests(unittest.TestCase):
    """Verify JSON conversion, validation, order, and path handling."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def valid_document(self) -> dict[str, object]:
        return {
            "worker": {
                "command": ["python", "workers/worker.py"],
                "metrics_argument": "--metrics-out",
                "timeout_seconds": 120.0,
            },
            "optimization": {
                "parameters": [
                    {
                        "name": "epochs",
                        "kind": "integer",
                        "minimum": 5,
                        "maximum": 100,
                    },
                    {
                        "name": "learning_rate",
                        "kind": "float",
                        "minimum": 0.0001,
                        "maximum": 0.1,
                    },
                    {
                        "name": "batch_size",
                        "kind": "categorical",
                        "choices": [8, 16, 32],
                    },
                ],
                "objectives": [
                    {
                        "metric_name": "validation_accuracy",
                        "direction": "maximize",
                    },
                    {
                        "metric_name": "validation_loss",
                        "direction": "minimize",
                    },
                ],
            },
            "algorithm": {"name": "random_search", "seed": 42},
            "stop_policy": {"max_trials": 20},
        }

    def write_document(
        self,
        document: object,
        name: str = "project.json",
    ) -> Path:
        path = self.root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_loads_complete_immutable_configuration(self) -> None:
        path = self.write_document(self.valid_document())

        configuration = load_configuration(path)

        self.assertEqual(configuration.worker.command[0], "python")
        self.assertEqual(
            configuration.worker.command[1],
            str((self.root / "workers/worker.py").resolve()),
        )
        self.assertEqual(configuration.algorithm.name, "random_search")
        self.assertEqual(configuration.algorithm.seed, 42)
        self.assertEqual(configuration.stop_policy.max_trials, 20)

    def test_accepts_any_registered_algorithm_name_not_just_random_search(
        self,
    ) -> None:
        # random_search is still the common default, but validation defers
        # to ALGORITHM_REGISTRY -- nsga2 is registered there too and must
        # be just as loadable from a config file.
        document = self.valid_document()
        document["algorithm"] = {"name": "nsga2", "seed": 7}

        configuration = load_configuration(self.write_document(document))

        self.assertEqual(configuration.algorithm.name, "nsga2")
        self.assertEqual(configuration.algorithm.seed, 7)

    def test_preserves_parameter_and_objective_order(self) -> None:
        configuration = load_configuration(
            self.write_document(self.valid_document())
        )

        self.assertEqual(
            tuple(
                parameter.name
                for parameter in configuration.optimization.parameters
            ),
            ("epochs", "learning_rate", "batch_size"),
        )
        self.assertEqual(
            tuple(
                parameter.kind
                for parameter in configuration.optimization.parameters
            ),
            (
                ParameterKind.INTEGER,
                ParameterKind.FLOAT,
                ParameterKind.CATEGORICAL,
            ),
        )
        self.assertEqual(
            tuple(
                objective.direction
                for objective in configuration.optimization.objectives
            ),
            (Direction.MAXIMIZE, Direction.MINIMIZE),
        )

    def test_loads_repository_iris_example(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        path = repository_root / "examples/iris_torch/iris_config.json"

        configuration = load_configuration(path)

        self.assertEqual(
            tuple(
                parameter.name
                for parameter in configuration.optimization.parameters
            ),
            ("learning_rate", "hidden_size", "epochs", "batch_size"),
        )
        self.assertEqual(
            configuration.worker.command,
            (
                "py",
                "-3.13",
                str((path.parent / "iris_worker.py").resolve()),
            ),
        )

    def test_loads_repository_synthetic_worker_example(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        path = (
            repository_root
            / "examples"
            / "zdt1_benchmark"
            / "synthetic_config.json"
        )

        configuration = load_configuration(path)

        self.assertEqual(
            configuration.worker.command,
            ("hyperloop-synthetic-worker",),
        )
        self.assertEqual(
            tuple(
                parameter.name
                for parameter in configuration.optimization.parameters
            ),
            ("x1", "x2", "x3", "x4"),
        )

    def test_reports_invalid_json_with_location(self) -> None:
        path = self.root / "invalid.json"
        path.write_text('{"worker": }', encoding="utf-8")

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(path)

        self.assertIn("line 1", str(caught.exception))
        self.assertIn("column", str(caught.exception))

    def test_rejects_duplicate_json_object_keys(self) -> None:
        path = self.root / "duplicate.json"
        path.write_text(
            '{"algorithm": {}, "algorithm": {}}',
            encoding="utf-8",
        )

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(path)

        self.assertIn("duplicate object key 'algorithm'", str(caught.exception))

    def test_reports_independent_root_contract_errors_together(self) -> None:
        document = self.valid_document()
        del document["algorithm"]
        document["unexpected"] = {}

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "root.algorithm: required field is missing",
            caught.exception.issues,
        )
        self.assertIn(
            "root.unexpected: field is not allowed",
            caught.exception.issues,
        )

    def test_reports_invalid_items_from_multiple_sections(self) -> None:
        document = self.valid_document()
        algorithm = document["algorithm"]
        stop_policy = document["stop_policy"]
        optimization = document["optimization"]
        assert isinstance(algorithm, dict)
        assert isinstance(stop_policy, dict)
        assert isinstance(optimization, dict)
        algorithm["name"] = "unsupported"
        stop_policy["max_trials"] = 0
        parameters = optimization["parameters"]
        assert isinstance(parameters, list)
        first_parameter = parameters[0]
        assert isinstance(first_parameter, dict)
        first_parameter["minimum"] = 200

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        message = str(caught.exception)
        self.assertIn("minimum cannot exceed maximum", message)
        self.assertIn("algorithm.name: must be one of", message)
        self.assertIn("max_trials must be positive", message)

    def test_rejects_unknown_nested_fields(self) -> None:
        document = self.valid_document()
        worker = document["worker"]
        optimization = document["optimization"]
        assert isinstance(worker, dict)
        assert isinstance(optimization, dict)
        worker["working_directory"] = "."
        objectives = optimization["objectives"]
        assert isinstance(objectives, list)
        first_objective = objectives[0]
        assert isinstance(first_objective, dict)
        first_objective["weight"] = 0.5

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "worker.working_directory: field is not allowed",
            caught.exception.issues,
        )
        self.assertIn(
            "optimization.objectives[0].weight: field is not allowed",
            caught.exception.issues,
        )

    def test_rejects_missing_file(self) -> None:
        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.root / "missing.json")

        self.assertTrue(caught.exception.issues[0].startswith("path:"))

    def test_rejects_unresolvable_path(self) -> None:
        # A path containing a null byte fails during Path.resolve() itself,
        # exercising the ValueError branch of _configuration_path rather
        # than a missing-file OSError.
        with self.assertRaises(ConfigurationError) as caught:
            load_configuration("bad\x00path.json")

        self.assertTrue(caught.exception.issues[0].startswith("path:"))

    def test_configuration_error_requires_at_least_one_issue(self) -> None:
        with self.assertRaises(ValueError):
            ConfigurationError(())

    def test_rejects_non_object_root_document(self) -> None:
        path = self.write_document([])

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(path)

        self.assertEqual(
            caught.exception.issues,
            ("root: must be a JSON object",),
        )

    def test_rejects_worker_section_that_is_not_an_object(self) -> None:
        document = self.valid_document()
        document["worker"] = "python workers/worker.py"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "worker: must be a JSON object",
            caught.exception.issues,
        )

    def test_rejects_worker_command_that_is_not_an_array(self) -> None:
        document = self.valid_document()
        worker = document["worker"]
        assert isinstance(worker, dict)
        worker["command"] = "python workers/worker.py"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "worker.command: must be a JSON array",
            caught.exception.issues,
        )

    def test_rejects_worker_command_with_non_string_items(self) -> None:
        document = self.valid_document()
        worker = document["worker"]
        assert isinstance(worker, dict)
        worker["command"] = ["python", 5]

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "worker.command: every item must be a string",
            caught.exception.issues,
        )

    def test_rejects_optimization_section_that_is_not_an_object(self) -> None:
        document = self.valid_document()
        document["optimization"] = []

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization: must be a JSON object",
            caught.exception.issues,
        )

    def test_rejects_optimization_section_with_missing_or_extra_fields(
        self,
    ) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        del optimization["objectives"]
        optimization["notes"] = "unexpected"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.objectives: required field is missing",
            caught.exception.issues,
        )
        self.assertIn(
            "optimization.notes: field is not allowed",
            caught.exception.issues,
        )

    def test_rejects_optimization_parameters_and_objectives_that_are_not_arrays(
        self,
    ) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        optimization["parameters"] = {}
        optimization["objectives"] = "validation_accuracy"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.parameters: must be a JSON array",
            caught.exception.issues,
        )
        self.assertIn(
            "optimization.objectives: must be a JSON array",
            caught.exception.issues,
        )

    def test_rejects_parameter_item_that_is_not_an_object(self) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        parameters = optimization["parameters"]
        assert isinstance(parameters, list)
        parameters[0] = "epochs"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.parameters[0]: must be a JSON object",
            caught.exception.issues,
        )

    def test_rejects_parameter_with_missing_or_extra_base_fields(
        self,
    ) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        parameters = optimization["parameters"]
        assert isinstance(parameters, list)
        parameters[0] = {"name": "epochs", "extra": True}

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.parameters[0].kind: required field is missing",
            caught.exception.issues,
        )
        self.assertIn(
            "optimization.parameters[0].extra: field is not allowed",
            caught.exception.issues,
        )

    def test_rejects_parameter_with_invalid_kind(self) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        parameters = optimization["parameters"]
        assert isinstance(parameters, list)
        first_parameter = parameters[0]
        assert isinstance(first_parameter, dict)
        first_parameter["kind"] = "bogus"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.parameters[0].kind: must be integer, float, "
            "or categorical",
            caught.exception.issues,
        )

    def test_rejects_numeric_parameter_missing_minimum_and_maximum(
        self,
    ) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        parameters = optimization["parameters"]
        assert isinstance(parameters, list)
        parameters[0] = {"name": "epochs", "kind": "integer"}

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.parameters[0].minimum: required field is missing",
            caught.exception.issues,
        )
        self.assertIn(
            "optimization.parameters[0].maximum: required field is missing",
            caught.exception.issues,
        )

    def test_rejects_categorical_choices_that_are_not_an_array(self) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        parameters = optimization["parameters"]
        assert isinstance(parameters, list)
        categorical_parameter = parameters[2]
        assert isinstance(categorical_parameter, dict)
        categorical_parameter["choices"] = "8,16,32"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.parameters[2].choices: must be a JSON array",
            caught.exception.issues,
        )

    def test_rejects_objective_item_that_is_not_an_object(self) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        objectives = optimization["objectives"]
        assert isinstance(objectives, list)
        objectives[0] = "validation_accuracy"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.objectives[0]: must be a JSON object",
            caught.exception.issues,
        )

    def test_rejects_objective_with_invalid_direction(self) -> None:
        document = self.valid_document()
        optimization = document["optimization"]
        assert isinstance(optimization, dict)
        objectives = optimization["objectives"]
        assert isinstance(objectives, list)
        first_objective = objectives[0]
        assert isinstance(first_objective, dict)
        first_objective["direction"] = "up"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "optimization.objectives[0].direction: must be minimize or "
            "maximize",
            caught.exception.issues,
        )

    def test_rejects_algorithm_section_that_is_not_an_object(self) -> None:
        document = self.valid_document()
        document["algorithm"] = []

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "algorithm: must be a JSON object",
            caught.exception.issues,
        )

    def test_rejects_algorithm_section_with_missing_or_extra_fields(
        self,
    ) -> None:
        document = self.valid_document()
        algorithm = document["algorithm"]
        assert isinstance(algorithm, dict)
        del algorithm["seed"]
        algorithm["notes"] = "unexpected"

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "algorithm.seed: required field is missing",
            caught.exception.issues,
        )
        self.assertIn(
            "algorithm.notes: field is not allowed",
            caught.exception.issues,
        )

    def test_rejects_stop_policy_section_that_is_not_an_object(self) -> None:
        document = self.valid_document()
        document["stop_policy"] = []

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "stop_policy: must be a JSON object",
            caught.exception.issues,
        )

    def test_rejects_stop_policy_section_with_missing_or_extra_fields(
        self,
    ) -> None:
        document = self.valid_document()
        document["stop_policy"] = {"notes": "unexpected"}

        with self.assertRaises(ConfigurationError) as caught:
            load_configuration(self.write_document(document))

        self.assertIn(
            "stop_policy.max_trials: required field is missing",
            caught.exception.issues,
        )
        self.assertIn(
            "stop_policy.notes: field is not allowed",
            caught.exception.issues,
        )


if __name__ == "__main__":
    unittest.main()
