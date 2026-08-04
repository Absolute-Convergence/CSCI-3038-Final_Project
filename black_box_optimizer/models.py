"""Immutable configuration and candidate models for the optimizer foundation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


ParameterValue = int | float | str
NumericBound = int | float
_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ParameterKind(StrEnum):
    """Supported parameter-domain kinds."""

    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class Direction(StrEnum):
    """Whether an objective is improved by smaller or larger values."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if not _is_number(value):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_parameter_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (int, str))


def _require_parameter_name(name: str) -> None:
    if not isinstance(name, str) or not _PARAMETER_NAME.fullmatch(name):
        raise ValueError(
            "parameter names must match [A-Za-z][A-Za-z0-9_]*"
        )


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """One named parameter and its legal domain."""

    name: str
    kind: ParameterKind
    minimum: NumericBound | None = None
    maximum: NumericBound | None = None
    choices: tuple[ParameterValue, ...] = ()

    def __post_init__(self) -> None:
        _require_parameter_name(self.name)
        if not isinstance(self.kind, ParameterKind):
            raise TypeError("kind must be a ParameterKind")

        copied_choices = tuple(self.choices)
        object.__setattr__(self, "choices", copied_choices)

        if self.kind is ParameterKind.INTEGER:
            self._validate_integer_domain()
        elif self.kind is ParameterKind.FLOAT:
            self._validate_float_domain()
        else:
            self._validate_categorical_domain()

    def _validate_integer_domain(self) -> None:
        bounds = (self.minimum, self.maximum)
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in bounds):
            raise ValueError("integer parameters require integer bounds")
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if self.choices:
            raise ValueError("integer parameters cannot declare choices")

    def _validate_float_domain(self) -> None:
        bounds = (self.minimum, self.maximum)
        if not all(_is_number(value) for value in bounds):
            raise ValueError("float parameters require numeric bounds")
        if not all(_is_finite_number(value) for value in bounds):
            raise ValueError("float parameter bounds must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("float minimum must be less than maximum")
        if self.choices:
            raise ValueError("float parameters cannot declare choices")

    def _validate_categorical_domain(self) -> None:
        if self.minimum is not None or self.maximum is not None:
            raise ValueError("categorical parameters cannot declare bounds")
        if not self.choices:
            raise ValueError("categorical parameters require choices")
        if not all(_is_parameter_value(value) for value in self.choices):
            raise ValueError(
                "categorical choices must be finite int, float, or str values"
            )
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("categorical choices must be unique")


@dataclass(frozen=True, slots=True)
class Objective:
    """One exact worker metric name and its optimization direction."""

    metric_name: str
    direction: Direction

    def __post_init__(self) -> None:
        is_nonempty_string = (
            isinstance(self.metric_name, str) and bool(self.metric_name.strip())
        )
        if not is_nonempty_string:
            raise ValueError("metric_name must be a nonempty string")
        if self.metric_name != self.metric_name.strip():
            raise ValueError("metric_name cannot have surrounding whitespace")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction")


@dataclass(frozen=True, slots=True)
class OptimizationContract:
    """Ordered parameter space and multi-objective definition."""

    parameters: tuple[ParameterDefinition, ...]
    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        objectives = tuple(self.objectives)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "objectives", objectives)

        if not parameters:
            raise ValueError("an optimization contract needs a parameter")
        if len(objectives) < 2:
            raise ValueError("an optimization contract needs two objectives")
        if len({item.name for item in parameters}) != len(parameters):
            raise ValueError("parameter names must be unique")
        if len({item.metric_name for item in objectives}) != len(objectives):
            raise ValueError("objective metric names must be unique")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """How the future runner will invoke one external worker."""

    command: tuple[str, ...]
    metrics_argument: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        command = tuple(self.command)
        object.__setattr__(self, "command", command)
        has_only_nonempty_strings = all(
            isinstance(part, str) and bool(part) for part in command
        )
        if not command or not has_only_nonempty_strings:
            raise ValueError("command must contain nonempty strings")
        if not isinstance(self.metrics_argument, str):
            raise ValueError("metrics_argument must be a string")
        if not self.metrics_argument.startswith("--"):
            raise ValueError("metrics_argument must start with --")
        if len(self.metrics_argument) == 2:
            raise ValueError("metrics_argument must include a flag name")
        if not _is_number(self.timeout_seconds):
            raise ValueError("timeout_seconds must be numeric")
        if not _is_finite_number(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Immutable algorithm selection settings for the MVP."""

    name: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("algorithm name must be a nonempty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")


@dataclass(frozen=True, slots=True)
class StopPolicy:
    """Maximum number of worker attempts authorized for one run."""

    max_trials: int

    def __post_init__(self) -> None:
        if isinstance(self.max_trials, bool) or not isinstance(
            self.max_trials, int
        ):
            raise ValueError("max_trials must be an integer")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")


@dataclass(frozen=True, slots=True)
class ProjectConfiguration:
    """Complete immutable project configuration."""

    worker: WorkerSpec
    optimization: OptimizationContract
    algorithm: AlgorithmSpec
    stop_policy: StopPolicy


@dataclass(frozen=True, slots=True)
class CandidateConfiguration:
    """One parameter mapping copied into a read-only view."""

    parameters: Mapping[str, ParameterValue]

    def __post_init__(self) -> None:
        copied_parameters = dict(self.parameters)
        if not copied_parameters:
            raise ValueError("a candidate needs at least one parameter")
        for name, value in copied_parameters.items():
            _require_parameter_name(name)
            if not _is_parameter_value(value):
                raise ValueError(
                    "candidate values must be finite int, float, or str values"
                )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(copied_parameters),
        )
