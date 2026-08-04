"""Validate proposed candidates against the declared optimization domain."""

from __future__ import annotations

import math

from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    ParameterValue,
)


class CandidateValidationError(ValueError):
    """Raised when a search proposal violates its declared parameter space."""


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def validate_candidate(
    candidate: CandidateConfiguration,
    contract: OptimizationContract,
) -> CandidateConfiguration:
    """Validate a proposal and return it ordered by the declared parameters."""
    declared_names = tuple(item.name for item in contract.parameters)
    candidate_names = tuple(candidate.parameters)
    missing = tuple(
        name for name in declared_names if name not in candidate.parameters
    )
    extra = tuple(
        name for name in candidate_names if name not in declared_names
    )

    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing parameters: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected parameters: {', '.join(extra)}")
        raise CandidateValidationError("; ".join(details))

    ordered: dict[str, ParameterValue] = {}
    for definition in contract.parameters:
        value = candidate.parameters[definition.name]
        _validate_value(definition, value)
        ordered[definition.name] = value

    return CandidateConfiguration(parameters=ordered)


def _validate_value(
    definition: ParameterDefinition,
    value: ParameterValue,
) -> None:
    if definition.kind is ParameterKind.INTEGER:
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and definition.minimum <= value <= definition.maximum
        )
        expected = (
            f"an integer from {definition.minimum} through "
            f"{definition.maximum}"
        )
    elif definition.kind is ParameterKind.FLOAT:
        valid = (
            _is_finite_number(value)
            and definition.minimum <= value <= definition.maximum
        )
        expected = (
            f"a finite number from {definition.minimum} through "
            f"{definition.maximum}"
        )
    else:
        valid = any(
            type(value) is type(choice) and value == choice
            for choice in definition.choices
        )
        expected = f"one of {definition.choices!r}"

    if not valid:
        raise CandidateValidationError(
            f"parameter {definition.name!r} must be {expected}; "
            f"received {value!r}"
        )
