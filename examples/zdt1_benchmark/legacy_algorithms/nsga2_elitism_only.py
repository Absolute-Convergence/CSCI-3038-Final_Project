"""
nsga2_elitism_only.py

The middle step in NSGA2's evolution: elitism added, but polynomial
mutation not yet -- mutation is still the original hard-reset behavior.

Built from the current shipped black_box_optimizer/search/nsga2.py (its
elitism machinery -- _select_survivors, self._current_population_ids, the
combined survivor-pool breeding in propose() -- is unchanged here) with
_mutate() reverted to the pre-polynomial-mutation version: every mutated
parameter gets a fresh random draw, regardless of kind.

Not registered anywhere, not used by the real optimizer -- this exists
purely so examples/zdt1_benchmark/ can compare the algorithm's evolution
against itself, one step at a time, without touching any existing
project file.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
)
from black_box_optimizer.pareto import build_pareto_front, is_eligible
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.search.base import ProposalResult
from black_box_optimizer.search.random_search import (
    _finite_space_size,
    _sample_value,
    candidate_key,
)

_MAX_DUPLICATE_ATTEMPTS = 100
_MIN_POPULATION_SIZE = 4
_MAX_POPULATION_SIZE = 10


def _default_population_size(
    contract: OptimizationContract,
    finite_size: int | None,
) -> int:
    """Choose a practical population size for this contract."""
    population_size = max(
        _MIN_POPULATION_SIZE,
        min(_MAX_POPULATION_SIZE, 2 * len(contract.parameters)),
    )
    if finite_size is not None:
        population_size = min(population_size, finite_size)
    return population_size


def _default_mutation_rate(contract: OptimizationContract) -> float:
    """Choose a rate that mutates about one parameter per child."""
    return 1.0 / len(contract.parameters)


class _RankedIndividual(NamedTuple):
    """Keep one trial together with its rank and crowding distance."""

    record: TrialRecord
    rank: int
    crowding_distance: float


def _rank_population(
    generation: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> list[_RankedIndividual]:
    """Rank completed trials by Pareto tier and crowding distance."""
    eligible = [
        record for record in generation if is_eligible(record, contract)
    ]
    ineligible = [
        record for record in generation if not is_eligible(record, contract)
    ]

    tiers: list[tuple[TrialRecord, ...]] = []
    remaining = list(eligible)
    while remaining:
        tier_records = build_pareto_front(remaining, contract).records
        tiers.append(tier_records)

        tier_trial_ids = {record.trial_id for record in tier_records}
        remaining = [
            record
            for record in remaining
            if record.trial_id not in tier_trial_ids
        ]

    ranked: list[_RankedIndividual] = []
    for rank, tier_records in enumerate(tiers):
        crowding = _crowding_distances(tier_records, contract)
        for record in tier_records:
            ranked.append(
                _RankedIndividual(
                    record=record,
                    rank=rank,
                    crowding_distance=crowding[record.trial_id],
                )
            )

    worst_rank = len(tiers)
    for record in ineligible:
        ranked.append(
            _RankedIndividual(
                record=record, rank=worst_rank, crowding_distance=0.0
            )
        )

    return ranked


def _select_survivors(
    combined_records: Sequence[TrialRecord],
    contract: OptimizationContract,
    population_size: int,
) -> list[TrialRecord]:
    """Keep the best records from the combined parent and child pool.

    Lower Pareto rank wins, then higher crowding distance. This is the
    elitism step that lets strong parents survive into later generations.
    """
    ranked = _rank_population(combined_records, contract)
    ranked.sort(
        key=lambda individual: (
            individual.rank,
            -individual.crowding_distance,
        )
    )
    return [individual.record for individual in ranked[:population_size]]


def _crowding_distances(
    tier_records: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> dict[int, float]:
    """Measure how isolated each record is within one Pareto tier."""
    distances = {record.trial_id: 0.0 for record in tier_records}

    if len(tier_records) <= 2:
        for record in tier_records:
            distances[record.trial_id] = float("inf")
        return distances

    for objective in contract.objectives:
        sorted_tier = sorted(
            tier_records,
            key=lambda record: record.metrics[objective.metric_name],
        )
        minimum_value = sorted_tier[0].metrics[objective.metric_name]
        maximum_value = sorted_tier[-1].metrics[objective.metric_name]

        objective_range = maximum_value - minimum_value
        if objective_range == 0:
            continue

        distances[sorted_tier[0].trial_id] = float("inf")
        distances[sorted_tier[-1].trial_id] = float("inf")

        for index in range(1, len(sorted_tier) - 1):
            trial_id = sorted_tier[index].trial_id
            if distances[trial_id] == float("inf"):
                continue

            previous_value = sorted_tier[index - 1].metrics[
                objective.metric_name
            ]
            next_value = sorted_tier[index + 1].metrics[objective.metric_name]
            gap = (next_value - previous_value) / objective_range
            distances[trial_id] += gap

    return distances


def _tournament_select(
    ranked_population: Sequence[_RankedIndividual],
    generator: np.random.Generator,
) -> _RankedIndividual:
    """Choose one parent using a two-candidate tournament."""
    first_index, second_index = generator.choice(
        len(ranked_population), size=2, replace=False
    )
    first = ranked_population[first_index]
    second = ranked_population[second_index]

    if first.rank != second.rank:
        return first if first.rank < second.rank else second

    if first.crowding_distance != second.crowding_distance:
        return (
            first
            if first.crowding_distance > second.crowding_distance
            else second
        )

    return first


def _crossover(
    parent_a: TrialRecord,
    parent_b: TrialRecord,
    contract: OptimizationContract,
    generator: np.random.Generator,
) -> dict[str, object]:
    """Build a child by taking each parameter from either parent."""
    child_parameters: dict[str, object] = {}
    for parameter in contract.parameters:
        if generator.random() < 0.5:
            child_parameters[parameter.name] = parent_a.parameters[
                parameter.name
            ]
        else:
            child_parameters[parameter.name] = parent_b.parameters[
                parameter.name
            ]
    return child_parameters


def _mutate(
    parameters: dict[str, object],
    contract: OptimizationContract,
    generator: np.random.Generator,
    mutation_rate: float,
) -> dict[str, object]:
    """Randomly replace some parameter values with newly sampled values.

    The original hard-reset mutation, kept as-is for this variant: every
    mutated parameter gets a completely fresh random draw, regardless of
    its current value. Polynomial mutation is intentionally not present
    here -- that's what distinguishes this variant from the current
    shipped algorithm.
    """
    mutated = dict(parameters)
    for parameter in contract.parameters:
        if generator.random() < mutation_rate:
            mutated[parameter.name] = _sample_value(generator, parameter)
    return mutated


def _sample_new_random_candidate(
    contract: OptimizationContract,
    generator: np.random.Generator,
    forbidden_keys: set,
) -> CandidateConfiguration | None:
    """Create one unused random candidate for generation 0."""
    for _ in range(_MAX_DUPLICATE_ATTEMPTS):
        parameters = {
            definition.name: _sample_value(generator, definition)
            for definition in contract.parameters
        }
        candidate = CandidateConfiguration(parameters=parameters)
        key = candidate_key(contract, candidate)
        if key not in forbidden_keys:
            return candidate

    return None


def _breed_one_child(
    ranked_population: Sequence[_RankedIndividual],
    contract: OptimizationContract,
    generator: np.random.Generator,
    mutation_rate: float,
    forbidden_keys: set,
) -> CandidateConfiguration | None:
    """Breed one child that has not already been tried."""
    for _ in range(_MAX_DUPLICATE_ATTEMPTS):
        parent_a = _tournament_select(ranked_population, generator).record
        parent_b = _tournament_select(ranked_population, generator).record

        child_parameters = _crossover(parent_a, parent_b, contract, generator)
        child_parameters = _mutate(
            child_parameters, contract, generator, mutation_rate
        )

        candidate = CandidateConfiguration(parameters=child_parameters)
        key = candidate_key(contract, candidate)
        if key not in forbidden_keys:
            return candidate

    return None


class NSGA2ElitismOnly:
    """Elitism added, polynomial mutation not yet.

    Each new generation combines the surviving parent population with the
    newest offspring, ranks the combined pool, and keeps the best
    population_size individuals -- same elitism as the current shipped
    algorithm. Mutation is still the original hard reset.
    """

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed cannot be negative")

        self._generator = np.random.default_rng(seed)

        self._population_size: int | None = None
        self._mutation_rate: float | None = None

        self._pending_children: deque[CandidateConfiguration] = deque()

        # Elitism needs to remember which completed trials are still part
        # of the current parent population.
        self._current_population_ids: set[int] = set()

    def propose(
        self,
        contract: OptimizationContract,
        history: Sequence[TrialRecord],
    ) -> ProposalResult:
        """Return the next candidate, creating a generation when necessary."""
        finite_size = _finite_space_size(contract)

        if self._population_size is None:
            self._population_size = _default_population_size(
                contract, finite_size
            )
            self._mutation_rate = _default_mutation_rate(contract)

        population_size = self._population_size
        assert self._mutation_rate is not None

        if self._pending_children:
            return ProposalResult(
                status="candidate", candidate=self._pending_children.popleft()
            )

        attempted_keys = {
            candidate_key(
                contract, CandidateConfiguration(parameters=record.parameters)
            )
            for record in history
        }

        if finite_size is not None and len(attempted_keys) >= finite_size:
            return ProposalResult(status="search_exhausted")

        if not history:
            batch = self._breed_random_generation(
                contract, attempted_keys, population_size
            )
        else:
            latest_offspring = history[-population_size:]

            surviving_parents = [
                record
                for record in history
                if record.trial_id in self._current_population_ids
            ]
            combined_pool = surviving_parents + list(latest_offspring)
            parent_population = _select_survivors(
                combined_pool, contract, population_size
            )
            self._current_population_ids = {
                record.trial_id for record in parent_population
            }

            batch = self._breed_next_generation(
                contract,
                parent_population,
                attempted_keys,
                population_size,
                self._mutation_rate,
            )

        if batch is None:
            return ProposalResult(
                status="proposal_failed",
                reason=(
                    f"could not assemble a generation of {population_size} "
                    f"unique candidates after {_MAX_DUPLICATE_ATTEMPTS} "
                    "attempts per candidate"
                ),
            )

        self._pending_children.extend(batch)
        return ProposalResult(
            status="candidate", candidate=self._pending_children.popleft()
        )

    def _breed_random_generation(
        self,
        contract: OptimizationContract,
        attempted_keys: set,
        population_size: int,
    ) -> list[CandidateConfiguration] | None:
        """Create the random, unique candidates used for generation 0."""
        forbidden_keys = set(attempted_keys)

        batch: list[CandidateConfiguration] = []
        for _ in range(population_size):
            candidate = _sample_new_random_candidate(
                contract, self._generator, forbidden_keys
            )
            if candidate is None:
                return None

            batch.append(candidate)
            forbidden_keys.add(candidate_key(contract, candidate))

        return batch

    def _breed_next_generation(
        self,
        contract: OptimizationContract,
        parent_population: Sequence[TrialRecord],
        attempted_keys: set,
        population_size: int,
        mutation_rate: float,
    ) -> list[CandidateConfiguration] | None:
        """Rank the current survivors and breed the next generation."""
        ranked_population = _rank_population(parent_population, contract)

        forbidden_keys = set(attempted_keys)

        batch: list[CandidateConfiguration] = []
        for _ in range(population_size):
            child = _breed_one_child(
                ranked_population,
                contract,
                self._generator,
                mutation_rate,
                forbidden_keys,
            )
            if child is None:
                return None

            batch.append(child)
            forbidden_keys.add(candidate_key(contract, child))

        return batch
