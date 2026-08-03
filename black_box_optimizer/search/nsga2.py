"""
nsga2.py

An NSGA-II (Non-dominated Sorting Genetic Algorithm II) search algorithm.

This is an ADDITIONAL search option. It is registered next to
RandomSearch in search/registry.py, not used as a replacement for it.

It follows the same protocol RandomSearch already follows and is selected
by name in AlgorithmSpec.

============================================================================
THE LONG AND SKINNY
============================================================================

RandomSearch chooses a completely new random candidate every time propose()
is called. It does not care what happened during any previous trial.

NSGA-II works in generations instead:

    1. Create a batch of `population_size` candidates.

       The first generation is random because there are no completed
       trials to learn from yet.

    2. Let the controller run each candidate normally.

       This file does not change worker execution. Trials still run one at
       a time exactly as they do with RandomSearch.

    3. Once the whole generation has results, rank the candidates using
       Pareto dominance.

       This uses the same underlying idea as pareto.py: a candidate is
       better when it is no worse on every objective and strictly better
       on at least one.

    4. Prefer better-ranked candidates when choosing parents.

    5. Breed a new generation by mixing parent parameter values together
       and occasionally mutating them.

    6. Run the new generation and repeat.

So instead of guessing blindly for the entire run, each new generation is
influenced by the results of the generation before it.

============================================================================
WHY THIS CLASS HAS TO KEEP SOME INTERNAL STATE
============================================================================

The SearchAlgorithm protocol only asks for ONE candidate at a time:

    propose(contract, history)

That makes perfect sense for the controller. It asks for one candidate,
runs one trial, adds the result to history, and asks again.

NSGA-II is naturally batch-shaped, though. It ranks a whole completed
generation and then breeds a whole new generation from it.

That leaves us with a small interface mismatch:

    NSGA-II wants to produce a batch.
    The controller only wants one candidate.

The solution is self._pending_children.

Whenever a new generation is created, all of its candidates are placed in
that queue. propose() then returns them one at a time until the queue is
empty. Once it is empty, the algorithm knows it is time to create another
generation.

TrialHistory is still the real record of what actually happened.
self._pending_children only remembers candidates the algorithm has already
created but has not handed to the controller yet.

That queue is not persisted. If the program crashes, any children still
waiting in it are lost. This project does not currently persist the
internal state of any search algorithm, including RandomSearch, so this is
not a new problem introduced specifically by NSGA-II.
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

# Genetic operations can sometimes keep producing candidates that have
# already been tried. We need a stopping point so the algorithm cannot
# retry forever in a small or mostly exhausted search space.
#
# The project spec does not give us a required number, so this matches the
# same documented limit already used by RandomSearch.
_MAX_DUPLICATE_ATTEMPTS = 100

# NSGA-II needs enough candidates in each generation to have some actual
# variety, but a giant population would not make sense for this project.
#
# The example configurations often use small trial limits. If a generation
# had 20 candidates and max_trials was also 20, the algorithm would spend
# the entire run creating its first random generation and never get to do
# any evolution at all.
#
# Four gives us enough candidates to rank and breed. Ten keeps one
# generation from eating the entire trial budget.
_MIN_POPULATION_SIZE = 4
_MAX_POPULATION_SIZE = 10


def _default_population_size(
    contract: OptimizationContract,
    finite_size: int | None,
) -> int:
    """Choose a population size based on the number of parameters.

    The project spec does not define a population size because NSGA-II is
    outside the original MVP. This is our own design choice.

    The normal rule starts with two candidates per parameter, with a
    minimum of four and a maximum of ten.

    There is one important exception for small finite spaces.

    I found this by actually testing a contract with one categorical
    parameter and only two possible values. The normal minimum population
    size would be four, but it is obviously impossible to create four
    unique candidates from a search space containing only two candidates.

    Clamping the population size to finite_size prevents that situation.
    Without it, the first proposal would fail with proposal_failed even
    though the honest result is simply that the complete space contains
    fewer than four candidates.
    """
    population_size = max(
        _MIN_POPULATION_SIZE,
        min(_MAX_POPULATION_SIZE, 2 * len(contract.parameters)),
    )
    if finite_size is not None:
        population_size = min(population_size, finite_size)
    return population_size


def _default_mutation_rate(contract: OptimizationContract) -> float:
    """Choose the chance that any one parameter will mutate.

    A common genetic-algorithm starting point is to mutate roughly one
    parameter per child on average. Giving every parameter a probability
    of 1 / number_of_parameters gets us that behavior.

    This is another documented NSGA-II design choice, not a value required
    by the original project spec.
    """
    return 1.0 / len(contract.parameters)


class _RankedIndividual(NamedTuple):
    """Bundle a completed trial with its NSGA-II ranking information.

    This is only used inside this file to make tournament selection easier
    to read.

    rank:
        Lower is better. Rank 0 is the real Pareto front, meaning nobody
        else in the generation dominates that candidate. Rank 1 is the
        next layer, then rank 2, and so on.

        Failed or otherwise ineligible trials get one shared worst rank
        after every real rank.

    crowding_distance:
        Measures how different this candidate's objective values are from
        nearby candidates in the same rank.

        Higher is better because NSGA-II wants to preserve different
        tradeoffs instead of filling the population with nearly identical
        results.
    """

    record: TrialRecord
    rank: int
    crowding_distance: float


def _rank_population(
    generation: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> list[_RankedIndividual]:
    """Rank one completed generation by Pareto tier and crowding distance.

    This is the non-dominated sorting part of NSGA-II.

    We do not need to rewrite the project's dominance logic here.
    pareto.build_pareto_front() already knows how to find the records that
    are not dominated by anything else.

    We can reuse it repeatedly:

        rank 0 = the Pareto front of the entire generation
        rank 1 = the Pareto front after rank 0 is removed
        rank 2 = the Pareto front after ranks 0 and 1 are removed

    We continue peeling off layers until every eligible record has a rank.

    FAILURE HANDLING

    A TrialRecord is not guaranteed to contain usable objective metrics.
    A process can fail, time out, fail to launch, omit its metrics file, or
    write malformed metrics. Those are expected outcomes elsewhere in the
    project, not weird impossible states.

    We cannot compare a failed trial using Pareto dominance because it has
    no real objective values. We also should not silently remove it,
    because it still occupied one slot in this generation.

    Every ineligible trial therefore receives:

        rank = one rank worse than the last real Pareto tier
        crowding_distance = 0.0

    This makes failed trials lose against every successful trial during
    parent selection without inventing fake metric values or changing the
    original TrialRecord.

    Nothing returned here is written back into history. These ranks only
    exist for this one selection step.
    """
    eligible = [
        record for record in generation if is_eligible(record, contract)
    ]
    ineligible = [
        record for record in generation if not is_eligible(record, contract)
    ]

    # Find one Pareto layer, remove it, and repeat on whatever is left.
    # This lets us reuse the project's tested Pareto logic instead of
    # quietly creating a second version of the same comparison here.
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

    # Ineligible trials all share the rank immediately after the final real
    # tier. If every trial failed, there are no real tiers, so their rank is
    # 0. They are still tied for worst because there is nothing successful
    # for them to rank behind.
    worst_rank = len(tiers)
    for record in ineligible:
        ranked.append(
            _RankedIndividual(
                record=record, rank=worst_rank, crowding_distance=0.0
            )
        )

    return ranked


def _crowding_distances(
    tier_records: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> dict[int, float]:
    """Calculate crowding distance for every record in one Pareto tier.

    Rank tells us how good a candidate is compared with the rest of the
    generation. Crowding distance answers a different question:

        Does this candidate represent a distinct tradeoff, or is it packed
        into a cluster of almost identical results?

    For each objective, the records are sorted by that metric. The record
    at each end is a boundary candidate and gets infinite distance so the
    algorithm preserves the edges of the tradeoff range.

    Everyone between those boundaries receives the normalized gap between
    the neighboring metric values.

    Distances from every objective are added together. A larger total means
    the candidate is in a less crowded part of the tier.

    Objective direction does not matter here. Whether an objective is
    minimized or maximized mattered when ranks were assigned. At this point
    we only care how far apart the values are.
    """
    distances = {record.trial_id: 0.0 for record in tier_records}

    # With at most two records, everyone is already on the boundary. There
    # is no interior point whose neighboring distance could be measured.
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

        # The smallest and largest values define this tier's boundary for
        # the current objective, so both are always preserved.
        distances[sorted_tier[0].trial_id] = float("inf")
        distances[sorted_tier[-1].trial_id] = float("inf")

        objective_range = maximum_value - minimum_value
        if objective_range == 0:
            # Everyone has the same value for this objective, so it cannot
            # tell us anything about how spread out the candidates are.
            # It also cannot be used as a divisor.
            continue

        for index in range(1, len(sorted_tier) - 1):
            trial_id = sorted_tier[index].trial_id
            if distances[trial_id] == float("inf"):
                # This record was already a boundary on another objective.
                # Adding a finite value to infinity would not change it.
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

    Pick two random individuals and compare them:

        1. Lower rank wins.
        2. If the ranks match, higher crowding distance wins.
        3. If both values match, either one is equally reasonable.

    Rank keeps selection focused on good results. Crowding distance keeps
    the population from collapsing into one tiny area of the Pareto front.
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

    # At this point they are genuinely tied. Choosing the first one is no
    # less correct than flipping another coin.
    return first


def _crossover(
    parent_a: TrialRecord,
    parent_b: TrialRecord,
    contract: OptimizationContract,
    generator: np.random.Generator,
) -> dict[str, object]:
    """Mix two parents together one parameter at a time.

    This uses uniform crossover. For every parameter, the child has a
    50/50 chance of inheriting the value from either parent.

    There are more complicated crossover methods for numeric values, but
    this is the simplest one that works the same way for every parameter
    kind supported by the project.

    FLOAT and INTEGER values could technically be blended together.
    CATEGORICAL values cannot. There is no meaningful mathematical average
    between two category names.

    Choosing one parent's complete value works correctly for FLOAT,
    INTEGER, and CATEGORICAL parameters without separate crossover rules.
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


def _mutate(
    parameters: dict[str, object],
    contract: OptimizationContract,
    generator: np.random.Generator,
    mutation_rate: float,
) -> dict[str, object]:
    """Randomly replace some parameter values with newly sampled values.

    Crossover can only rearrange values that already exist in the parent
    population. Without mutation, a value that disappears from the
    population is gone forever, and a completely new value can never enter.

    Mutation keeps a controlled amount of randomness in the search so it
    can still explore new parts of the parameter space.

    This reuses random_search.py's _sample_value() function instead of
    creating another sampler here. That function already handles FLOAT,
    INTEGER, and CATEGORICAL parameters, including the NumPy object-dtype
    fix needed for mixed categorical values.
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
    """Create one random candidate that has not already been used.

    Generation 0 cannot be bred because there are no completed candidates
    to use as parents yet. It is seeded with random candidates instead.

    A sampled candidate may collide with one already in history or another
    candidate created for the same batch. We retry up to the documented
    duplicate-attempt limit and return None if no new candidate is found.
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
    """Select two parents and try to create one unused child.

    Crossover can easily recreate one of its parents, especially when the
    contract only contains a few parameters. Mutation can also randomly
    produce a combination that has already been tried.

    Because duplicate children are possible, breeding uses the same bounded
    retry strategy as random candidate sampling. It returns None if it
    cannot produce a new candidate within the allowed number of attempts.
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
    """An NSGA-II search algorithm using the project's existing protocol.

    The algorithm plans candidates in generations internally, but propose()
    still returns exactly one candidate at a time, just like RandomSearch.

    Construct it with a seed and call:

        propose(contract, history)

    Nothing about the controller's use of a SearchAlgorithm changes.
    """

    def __init__(self, seed: int) -> None:
        # Keep the validation consistent with RandomSearch so invalid seeds
        # produce a clear project-level error instead of a NumPy error.
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed cannot be negative")

        self._generator = np.random.default_rng(seed)

        # These depend on the OptimizationContract, which is not available
        # in __init__. They are calculated during the first propose() call
        # and reused afterward because the contract does not change mid-run.
        self._population_size: int | None = None
        self._mutation_rate: float | None = None

        # A generation is bred all at once, but the controller only asks for
        # one candidate at a time. This queue holds the rest of the already
        # created generation until the controller is ready for each one.
        self._pending_children: deque[CandidateConfiguration] = deque()

    def propose(
        self,
        contract: OptimizationContract,
        history: Sequence[TrialRecord],
    ) -> ProposalResult:
        """Return the next candidate, creating a generation when necessary."""
        # We need finite_size for the exhaustion check on every call. On the
        # first call, it also prevents the population from being larger than
        # the entire possible search space.
        finite_size = _finite_space_size(contract)

        if self._population_size is None:
            self._population_size = _default_population_size(
                contract, finite_size
            )
            self._mutation_rate = _default_mutation_rate(contract)

        population_size = self._population_size
        # Both are assigned together above.
        assert self._mutation_rate is not None

        # A generation has already been created and still has candidates
        # waiting. Return the next one instead of breeding another batch.
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

        # In a finite space, reaching every possible candidate means the
        # search is honestly finished. Genetic operations cannot invent a
        # legal combination outside the contract.
        if finite_size is not None and len(attempted_keys) >= finite_size:
            return ProposalResult(status="search_exhausted")

        if not history:
            # There are no completed trials to rank or use as parents yet,
            # so generation 0 has to be random.
            batch = self._breed_random_generation(
                contract, attempted_keys, population_size
            )
        else:
            # Every trial occupies one history slot whether it succeeded or
            # failed, so the most recent population_size records represent
            # the generation that just finished.
            generation = history[-population_size:]
            batch = self._breed_next_generation(
                contract,
                generation,
                attempted_keys,
                population_size,
                self._mutation_rate,
            )

        if batch is None:
            # The algorithm used all allowed retries without assembling a
            # complete batch of unique candidates. This follows the same
            # proposal_failed convention used by RandomSearch.
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
        # Use a copy because forbidden_keys also needs to include candidates
        # created during this batch. They are not in history yet, but they
        # still must not duplicate each other.
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
        generation: Sequence[TrialRecord],
        attempted_keys: set,
        population_size: int,
        mutation_rate: float,
    ) -> list[CandidateConfiguration] | None:
        """Rank one completed generation and breed its children."""
        ranked_population = _rank_population(generation, contract)

        # Children need to be unique against both history and the other
        # children being created for this same generation.
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