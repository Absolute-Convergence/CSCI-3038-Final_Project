"""Seeded RandomSearch, outlined in 7.2 - 7.5."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    ParameterValue,
)
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.search.base import ProposalResult

CandidateKey = tuple[tuple[str, ParameterValue], ...]

# The spec says duplicate retries need a limit but never picks one so
# Uses 100 so the search can't get stuck forever, but we can change this
# later if we think a different number is better
_MAX_DUPLICATE_ATTEMPTS = 100


def candidate_key(
    contract: OptimizationContract,
    candidate: CandidateConfiguration,
) -> CandidateKey:
    """Build the key we use to tell whether a candidate was already tried."""
    return tuple(
        (definition.name, candidate.parameters[definition.name])
        for definition in contract.parameters
    )


def _domain_size(definition: ParameterDefinition) -> int | None:
    """Return how many legal values this parameter has, if we can count them."""
    if definition.kind is ParameterKind.INTEGER:
        return definition.maximum - definition.minimum + 1
    if definition.kind is ParameterKind.CATEGORICAL:
        return len(definition.choices)

    # Float parameters aren't countable!
    return None


def _finite_space_size(contract: OptimizationContract) -> int | None:
    """Return the total search-space size, or None if it isn't finite."""
    size = 1

    for definition in contract.parameters:
        domain_size = _domain_size(definition)
        if domain_size is None:
            return None
        size *= domain_size

    return size


def _sample_value(
    generator: np.random.Generator,
    definition: ParameterDefinition,
) -> ParameterValue:
    """Sample one legal value for a single parameter."""
    if definition.kind is ParameterKind.INTEGER:
        sampled = generator.integers(
            definition.minimum,
            definition.maximum + 1,
        )
        return int(sampled)

    if definition.kind is ParameterKind.FLOAT:
        return float(
            generator.uniform(
                definition.minimum,
                definition.maximum,
            )
        )

    # I discovered NumPy will try to convert any mixed type choices into a
    # common type before sampling, so I used dtype=object in order to keep
    # each choice as the original Python value, huzzah!
    choices = np.asarray(definition.choices, dtype=object)
    return generator.choice(choices)


class RandomSearch:
    """The project's MVP search algorithm."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed cannot be negative")

        self._generator = np.random.default_rng(seed)

    def propose(
        self,
        contract: OptimizationContract,
        history: Sequence[TrialRecord],
    ) -> ProposalResult:
        """Try to find one new candidate to evaluate."""

        # So each previous attempt counts, even the failed or timed-out ones and
        # once we've tried a configuration, we shan't try it again quietly
        attempted_keys = {
            candidate_key(
                contract,
                CandidateConfiguration(parameters=record.parameters),
            )
            for record in history
        }

        finite_size = _finite_space_size(contract)
        if (
            finite_size is not None
            and len(attempted_keys) >= finite_size
        ):
            return ProposalResult(status="search_exhausted")

        for _ in range(_MAX_DUPLICATE_ATTEMPTS):
            parameters = {
                definition.name: _sample_value(
                    self._generator,
                    definition,
                )
                for definition in contract.parameters
            }

            candidate = CandidateConfiguration(parameters=parameters)
            key = candidate_key(contract, candidate)

            if key not in attempted_keys:
                return ProposalResult(
                    status="candidate",
                    candidate=candidate,
                )

        # If we keep landing on candidates we've tried before, this reports
        # we couldn't find a new one rather than looping forever, which
        # would be foolish
        return ProposalResult(
            status="proposal_failed",
            reason=(
                f"could not find an untried candidate after "
                f"{_MAX_DUPLICATE_ATTEMPTS} attempts"
            ),
        )