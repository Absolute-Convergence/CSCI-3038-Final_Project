"""
nsga2.py

An optional, updated, and totally Emilyfied NSGA-II search algorithm.

RandomSearch starts fresh every time propose() is called but NSGA2 learns
from completed trials (nice!) by working in generations:

    1. Build a generation of candidates
    2. Let the controller run them one at a time
    3. Rank the results using Pareto dominance
    4. Prefer stronger and less crowded candidates as parents
    5. Breed and mutate the next generation

The controller still asks for one singlet candidate at a time, so this
class keeps the rest of each generation in self._pending_children until
they're needed. TrialHistory remains the real record of completed work!

Our NSGA2 is now elitist, so the strong parents may survive alongside newer
offspring and the self._current_population_ids was created to remember
which completed trials are still part of the current parent population.
Neither piece of internal state is persisted, which totally matches the rest
of the project's search algorithm standardz!
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
    ParameterKind,
)
from black_box_optimizer.pareto import build_pareto_front, is_eligible
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.search.base import ProposalResult
from black_box_optimizer.search.random_search import (
    _finite_space_size,
    _sample_value,
    candidate_key,
)

# Breeding can keep rediscovering old candidates, so retries need a limit
# This matches the existing duplicate rule for random search
_MAX_DUPLICATE_ATTEMPTS = 100

# Four give enough variety to rank and breed and ten keeps
# one generation from swallowing a tiny trial budget whole
_MIN_POPULATION_SIZE = 4
_MAX_POPULATION_SIZE = 10


def _default_population_size(
    contract: OptimizationContract,
    finite_size: int | None,
) -> int:
    """Choose a practical population size for this contract.

    The default is two candidates per parameter, clamped between four and
    ten. Small finite spaces are clamped again so we never ask for more
    unique candidates than actually exist.
    """
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
    """Keep one trial together with its rank and crowding distance.

    Lower rank is better. Higher crowding distance is better within the
    same rank because it preserves a wider variety of tradeoffs.
    """

    record: TrialRecord
    rank: int
    crowding_distance: float


def _rank_population(
    generation: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> list[_RankedIndividual]:
    """Rank completed trials by Pareto tier and crowding distance.

    We repeatedly peel off Pareto fronts using the project's existing
    dominance logic. Failed or otherwise ineligible trials cannot be fairly
    compared, so they share one worst rank with zero crowding distance.
    These temporary rankings are never written back into history.
    """
    eligible = [
        record for record in generation if is_eligible(record, contract)
    ]
    ineligible = [
        record for record in generation if not is_eligible(record, contract)
    ]

    # Peel off one Pareto layer at a time instead of rewriting dominance
    # Like an onion
    # ...or an ogre
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

    # Failed trials share the rank immediately after the final real tier.
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
    """Measure how isolated each record is within one Pareto tier.

    Boundary records get infinite distance so the edges survive. Interior
    records collect normalized gaps from each objective. Higher distance
    means a less crowded and therefore more useful tradeoff.
    """
    distances = {record.trial_id: 0.0 for record in tier_records}

    # With two or fewer records, everybody is already on the boundary.
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
            # A tied objective adds no spacing information and cannot divide
            continue

        # This preserves both ends of this objectives tradeoff range
        distances[sorted_tier[0].trial_id] = float("inf")
        distances[sorted_tier[-1].trial_id] = float("inf")

        for index in range(1, len(sorted_tier) - 1):
            trial_id = sorted_tier[index].trial_id
            if distances[trial_id] == float("inf"):
                # Its already a boundary somewhere else, so infinity wins.
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
    """Choose one parent using a two-candidate tournament.

    Lower rank wins. Ties go to higher crowding distance, which keeps the
    search from collapsing into one tiny corner of the Pareto front.
    """
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

    # Fully tied, so the first one wins no need to get fancy
    return first


def _crossover(
    parent_a: TrialRecord,
    parent_b: TrialRecord,
    contract: OptimizationContract,
    generator: np.random.Generator,
) -> dict[str, object]:
    """Build a child by taking each parameter from either parent.

    Uniform crossover is simple and works for numeric and categorical
    parameters without separate rules.
    """
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


# Higher values keep polynomial mutations closer to the original value
# Twenty is the standard default for this algo they say
_POLYNOMIAL_MUTATION_ETA = 20.0


def _polynomial_mutate_value(
    value: object,
    parameter,
    generator: np.random.Generator,
) -> object:
    """Nudge one numeric value without fully replacing it.

    Polynomial mutation stays inside the parameter bounds. INTEGER values
    are rounded and clamped again afterward.
    """
    lower = float(parameter.minimum)
    upper = float(parameter.maximum)
    if upper == lower:
        return value

    x = float(value)
    delta1 = (x - lower) / (upper - lower)
    delta2 = (upper - x) / (upper - lower)
    u = generator.random()
    mutation_power = 1.0 / (_POLYNOMIAL_MUTATION_ETA + 1.0)

    if u <= 0.5:
        xy = 1.0 - delta1
        val = 2.0 * u + (1.0 - 2.0 * u) * (
            xy ** (_POLYNOMIAL_MUTATION_ETA + 1.0)
        )
        delta_q = val**mutation_power - 1.0
    else:
        xy = 1.0 - delta2
        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (
            xy ** (_POLYNOMIAL_MUTATION_ETA + 1.0)
        )
        delta_q = 1.0 - val**mutation_power

    mutated_value = min(max(x + delta_q * (upper - lower), lower), upper)

    if parameter.kind == ParameterKind.INTEGER:
        mutated_value = min(max(round(mutated_value), int(lower)), int(upper))
        return int(mutated_value)
    return mutated_value


def _mutate(
    parameters: dict[str, object],
    contract: OptimizationContract,
    generator: np.random.Generator,
    mutation_rate: float,
) -> dict[str, object]:
    """Mutate some parameters so the search can still discover new values.

    Numeric values get a nearby polynomial mutation. Categories have no
    meaningful notion of nearby, so they receive a fresh random draw.
    """
    mutated = dict(parameters)
    for parameter in contract.parameters:
        if generator.random() < mutation_rate:
            if parameter.kind == ParameterKind.CATEGORICAL:
                mutated[parameter.name] = _sample_value(generator, parameter)
            else:
                mutated[parameter.name] = _polynomial_mutate_value(
                    mutated[parameter.name], parameter, generator
                )
    return mutated


def _sample_new_random_candidate(
    contract: OptimizationContract,
    generator: np.random.Generator,
    forbidden_keys: set,
) -> CandidateConfiguration | None:
    """Create one unused random candidate for generation 0.

    Retry boundedly when a sample collides with history or this batch.
    """
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
    """Breed one child that has not already been tried.

    Crossover and mutation can still create duplicates, so retries are
    bounded just like random sampling.
    """
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


class NSGA2:
    """NSGA-II adapted to the project's one-candidate-at-a-time protocol."""

    def __init__(self, seed: int) -> None:
        # Match RandomSearch's validation and keep NumPy errors out of sight.
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed cannot be negative")

        self._generator = np.random.default_rng(seed)

        # These need the contract, so calculate them on the first proposal
        self._population_size: int | None = None
        self._mutation_rate: float | None = None

        # The controller wants one singlet child at a time so the rest wait here
        self._pending_children: deque[CandidateConfiguration] = deque()

        # Elitism can keep older winners around so remember their trial IDs
        self._current_population_ids: set[int] = set()

    def propose(
        self,
        contract: OptimizationContract,
        history: Sequence[TrialRecord],
    ) -> ProposalResult:
        """Return the next candidate, creating a generation when necessary."""
        # finite_size handles exhaustion and caps tiny search spaces
        finite_size = _finite_space_size(contract)

        if self._population_size is None:
            self._population_size = _default_population_size(
                contract, finite_size
            )
            self._mutation_rate = _default_mutation_rate(contract)

        population_size = self._population_size
        # Both are assigned together above
        assert self._mutation_rate is not None

        # Finish handing out the current generation before breeding another
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

        # Genetic operations cannot invent candidates outside the contract
        if finite_size is not None and len(attempted_keys) >= finite_size:
            return ProposalResult(status="search_exhausted")

        if not history:
            # No parents yet, so generation 0 has gots to be random
            batch = self._breed_random_generation(
                contract, attempted_keys, population_size
            )
        else:
            # The newest population_size records are this round's offspring
            # including any failed trials that still occupied a slot
            latest_offspring = history[-population_size:]

            # Let old winners compete with the newest offspring then keep
            # only the strongest population_size records
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
            # We ran out of duplicate retries before finishing the batch
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
        # Include this batch too since its candidates aren't in history yet
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

        # Children must! be unique against both history! and this new batch!
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

    # Tadaaaa! Emily's fancy algorithm!