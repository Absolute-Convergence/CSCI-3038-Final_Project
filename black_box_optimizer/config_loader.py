"""Load and validate immutable optimizer configuration from JSON."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypeVar

from black_box_optimizer.models import (
    AlgorithmSpec,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    ProjectConfiguration,
    StopPolicy,
    WorkerSpec,
)
from black_box_optimizer.search.registry import ALGORITHM_REGISTRY


JsonObject = dict[str, object]
_ModelT = TypeVar("_ModelT")
_PATH_SUFFIXES = frozenset(
    {".bat", ".cmd", ".exe", ".ps1", ".py", ".pyw", ".sh"}
)


class ConfigurationError(ValueError):
    """One or more errors in a project configuration."""

    def __init__(self, issues: Iterable[str]) -> None:
        copied_issues = tuple(issues)
        if not copied_issues:
            raise ValueError("ConfigurationError requires at least one issue")
        self.issues = copied_issues
        details = "\n".join(f"- {issue}" for issue in copied_issues)
        super().__init__(f"invalid project configuration:\n{details}")


class _DuplicateKeyError(ValueError):
    """Internal signal raised while decoding an object with duplicate keys."""


def load_configuration(path: str | Path) -> ProjectConfiguration:
    """Load one JSON file into a validated immutable configuration."""

    configuration_path = _configuration_path(path)
    document = _read_json(configuration_path)
    if not isinstance(document, dict):
        raise ConfigurationError(("root: must be a JSON object",))

    issues: list[str] = []
    _check_keys(
        document,
        required={"algorithm", "optimization", "stop_policy", "worker"},
        allowed={"algorithm", "optimization", "stop_policy", "worker"},
        location="root",
        issues=issues,
    )

    worker = _build_present_section(
        document,
        "worker",
        issues,
        lambda value: _build_worker(
            value,
            configuration_path.parent,
            issues,
        ),
    )
    optimization = _build_present_section(
        document,
        "optimization",
        issues,
        lambda value: _build_optimization(value, issues),
    )
    algorithm = _build_present_section(
        document,
        "algorithm",
        issues,
        lambda value: _build_algorithm(value, issues),
    )
    stop_policy = _build_present_section(
        document,
        "stop_policy",
        issues,
        lambda value: _build_stop_policy(value, issues),
    )

    if issues:
        raise ConfigurationError(issues)

    assert worker is not None
    assert optimization is not None
    assert algorithm is not None
    assert stop_policy is not None
    return ProjectConfiguration(
        worker=worker,
        optimization=optimization,
        algorithm=algorithm,
        stop_policy=stop_policy,
    )


def _configuration_path(path: str | Path) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, TypeError, ValueError) as error:
        raise ConfigurationError((f"path: {error}",)) from error


def _read_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(
                stream,
                object_pairs_hook=_object_without_duplicates,
            )
    except _DuplicateKeyError as error:
        raise ConfigurationError((f"JSON: {error}",)) from error
    except json.JSONDecodeError as error:
        issue = (
            f"JSON: {error.msg} at line {error.lineno}, "
            f"column {error.colno}"
        )
        raise ConfigurationError((issue,)) from error
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigurationError((f"path: {error}",)) from error


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _build_present_section(
    document: JsonObject,
    name: str,
    issues: list[str],
    builder: Callable[[object], _ModelT | None],
) -> _ModelT | None:
    if name not in document:
        return None
    return builder(document[name])


def _build_worker(
    value: object,
    configuration_directory: Path,
    issues: list[str],
) -> WorkerSpec | None:
    section = _require_object(value, "worker", issues)
    if section is None:
        return None
    required = {"command", "metrics_argument", "timeout_seconds"}
    if not _check_keys(
        section,
        required=required,
        allowed=required,
        location="worker",
        issues=issues,
    ):
        return None

    command = section["command"]
    if not isinstance(command, list):
        issues.append("worker.command: must be a JSON array")
        return None
    if not all(isinstance(part, str) for part in command):
        issues.append("worker.command: every item must be a string")
        return None

    resolved_parts: list[str] = []
    for index, part in enumerate(command):
        try:
            resolved_parts.append(
                _resolve_command_part(part, configuration_directory)
            )
        except (OSError, ValueError) as error:
            issues.append(f"worker.command[{index}]: {error}")
    if len(resolved_parts) != len(command):
        return None
    resolved_command = tuple(resolved_parts)
    metrics_argument = section["metrics_argument"]
    timeout_seconds = section["timeout_seconds"]
    return _construct(
        "worker",
        issues,
        lambda: WorkerSpec(
            command=resolved_command,
            metrics_argument=metrics_argument,  # type: ignore[arg-type]
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
        ),
    )


def _resolve_command_part(part: str, configuration_directory: Path) -> str:
    path = Path(part)
    foreign_absolute = (
        PureWindowsPath(part).is_absolute()
        or PurePosixPath(part).is_absolute()
    ) and not path.is_absolute()
    if foreign_absolute:
        raise ValueError(
            "absolute path uses syntax for a different operating system"
        )
    is_path = (
        path.is_absolute()
        or len(path.parts) > 1
        or path.suffix.lower() in _PATH_SUFFIXES
    )
    if not is_path or path.is_absolute() or part.startswith("-"):
        return part
    return str((configuration_directory / path).resolve())


def _build_optimization(
    value: object,
    issues: list[str],
) -> OptimizationContract | None:
    section = _require_object(value, "optimization", issues)
    if section is None:
        return None
    required = {"objectives", "parameters"}
    if not _check_keys(
        section,
        required=required,
        allowed=required,
        location="optimization",
        issues=issues,
    ):
        return None

    raw_parameters = _require_array(
        section["parameters"],
        "optimization.parameters",
        issues,
    )
    raw_objectives = _require_array(
        section["objectives"],
        "optimization.objectives",
        issues,
    )
    if raw_parameters is None or raw_objectives is None:
        return None

    parameters = tuple(
        parameter
        for index, item in enumerate(raw_parameters)
        if (
            parameter := _build_parameter(
                item,
                f"optimization.parameters[{index}]",
                issues,
            )
        )
        is not None
    )
    objectives = tuple(
        objective
        for index, item in enumerate(raw_objectives)
        if (
            objective := _build_objective(
                item,
                f"optimization.objectives[{index}]",
                issues,
            )
        )
        is not None
    )
    if (
        len(parameters) != len(raw_parameters)
        or len(objectives) != len(raw_objectives)
    ):
        return None
    return _construct(
        "optimization",
        issues,
        lambda: OptimizationContract(parameters, objectives),
    )


def _build_parameter(
    value: object,
    location: str,
    issues: list[str],
) -> ParameterDefinition | None:
    item = _require_object(value, location, issues)
    if item is None:
        return None

    base_keys = {"kind", "name"}
    all_keys = base_keys | {"choices", "maximum", "minimum"}
    if not _check_keys(
        item,
        required=base_keys,
        allowed=all_keys,
        location=location,
        issues=issues,
    ):
        return None

    try:
        kind = ParameterKind(item["kind"])
    except (TypeError, ValueError):
        issues.append(
            f"{location}.kind: must be integer, float, or categorical"
        )
        return None

    if kind is ParameterKind.CATEGORICAL:
        required = base_keys | {"choices"}
        allowed = required
    else:
        required = base_keys | {"maximum", "minimum"}
        allowed = required
    if not _check_keys(
        item,
        required=required,
        allowed=allowed,
        location=location,
        issues=issues,
    ):
        return None

    choices: object = item.get("choices", ())
    if kind is ParameterKind.CATEGORICAL:
        if not isinstance(choices, list):
            issues.append(f"{location}.choices: must be a JSON array")
            return None
        choices = tuple(choices)

    return _construct(
        location,
        issues,
        lambda: ParameterDefinition(
            name=item["name"],  # type: ignore[arg-type]
            kind=kind,
            minimum=item.get("minimum"),  # type: ignore[arg-type]
            maximum=item.get("maximum"),  # type: ignore[arg-type]
            choices=choices,  # type: ignore[arg-type]
        ),
    )


def _build_objective(
    value: object,
    location: str,
    issues: list[str],
) -> Objective | None:
    item = _require_object(value, location, issues)
    if item is None:
        return None
    required = {"direction", "metric_name"}
    if not _check_keys(
        item,
        required=required,
        allowed=required,
        location=location,
        issues=issues,
    ):
        return None
    try:
        direction = Direction(item["direction"])
    except (TypeError, ValueError):
        issues.append(f"{location}.direction: must be minimize or maximize")
        return None
    return _construct(
        location,
        issues,
        lambda: Objective(
            metric_name=item["metric_name"],  # type: ignore[arg-type]
            direction=direction,
        ),
    )


def _build_algorithm(
    value: object,
    issues: list[str],
) -> AlgorithmSpec | None:
    section = _require_object(value, "algorithm", issues)
    if section is None:
        return None
    required = {"name", "seed"}
    if not _check_keys(
        section,
        required=required,
        allowed=required,
        location="algorithm",
        issues=issues,
    ):
        return None
    if section["name"] not in ALGORITHM_REGISTRY:
        # random_search remains the well-known default most configs will
        # name, but validation defers to the registry rather than
        # hardcoding one algorithm -- registering a new SearchAlgorithm in
        # ALGORITHM_REGISTRY is enough to make it choosable from a config
        # file too, with no second list to keep in sync.
        valid = sorted(ALGORITHM_REGISTRY)
        issues.append(f"algorithm.name: must be one of {valid}")
        return None
    return _construct(
        "algorithm",
        issues,
        lambda: AlgorithmSpec(
            name=section["name"],  # type: ignore[arg-type]
            seed=section["seed"],  # type: ignore[arg-type]
        ),
    )


def _build_stop_policy(
    value: object,
    issues: list[str],
) -> StopPolicy | None:
    section = _require_object(value, "stop_policy", issues)
    if section is None:
        return None
    required = {"max_trials"}
    if not _check_keys(
        section,
        required=required,
        allowed=required,
        location="stop_policy",
        issues=issues,
    ):
        return None
    return _construct(
        "stop_policy",
        issues,
        lambda: StopPolicy(
            max_trials=section["max_trials"],  # type: ignore[arg-type]
        ),
    )


def _require_object(
    value: object,
    location: str,
    issues: list[str],
) -> JsonObject | None:
    if not isinstance(value, dict):
        issues.append(f"{location}: must be a JSON object")
        return None
    return value


def _require_array(
    value: object,
    location: str,
    issues: list[str],
) -> list[object] | None:
    if not isinstance(value, list):
        issues.append(f"{location}: must be a JSON array")
        return None
    return value


def _check_keys(
    value: JsonObject,
    *,
    required: set[str],
    allowed: set[str],
    location: str,
    issues: list[str],
) -> bool:
    issue_count = len(issues)
    for key in sorted(required - value.keys()):
        issues.append(f"{location}.{key}: required field is missing")
    for key in sorted(value.keys() - allowed):
        issues.append(f"{location}.{key}: field is not allowed")
    return len(issues) == issue_count


def _construct(
    location: str,
    issues: list[str],
    factory: Callable[[], _ModelT],
) -> _ModelT | None:
    try:
        return factory()
    except (TypeError, ValueError, OverflowError) as error:
        issues.append(f"{location}: {error}")
        return None
