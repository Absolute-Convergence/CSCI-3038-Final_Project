# NSGA-II Search Algorithm Notes

- **Status:** personal design notes
- **Author:** Emily Tew <3
- **Scope:** an optional `SearchAlgorithm` implementation registered
  alongside `RandomSearch`

These notes started as a way to sanity-check the design before I wrote any
code. By the end they also became a record of what actually shipped, what
changed during implementation, and what I think could still improve later.

---

# Why include NSGA-II?

The `search/registry.py` file was built around a registry of algorithms by
name, allowing additional search algorithms to be registered without
changing the controller. Random search is also intentionally simple, and 
was meant to be included for the MVP version of the pipeline. 

It is a totally great baseline search algorithm, but this project is 
fundamentally about multi-objective optimization and Pareto fronts. If 
there's any algorithm that naturally belongs beside random search, it's 
NSGA-II. Instead of treating every proposal as another independent guess, 
the addition of NSGA-II means we can use completed trials to influence what
comes next.

---

# The Big Idea

The existing `SearchAlgorithm` interface is:

```python
propose(contract, history)
```

The controller expects exactly one candidate per call. It does not know or
care how that candidate was chosen.

NSGA-II, however, naturally operates on whole generations rather than
individual candidates.

```text
Controller
    |
    | propose(contract, history)
    v

NSGA2
    |
    | generates a full generation
    v

pending_children queue
    |
    | one candidate returned per call
    v

Controller
```
 
To keep the existing `SearchAlgorithm` interface unchanged, the
implementation bridges this mismatch with a small in-memory queue:

```python
self._pending_children
```

When a new generation is bred, every child is placed into the queue.

Each subsequent `propose()` call returns one queued candidate. Once the
queue is empty, the next generation is bred.

This queue is the only algorithm-specific state maintained between calls.
Everything else is reconstructed from `TrialHistory`.

---

# Major design decisions

## Failed trials

This ended up being the part I spent the most time pondering crazystyle.

A `TrialRecord` isn't guaranteed to contain usable objective values.

So, a worker can:

- fail to launch
- crash
- time out
- never produce metrics
- produce malformed metrics
- produce non-finite metrics

Since our project treats those as normal outcomes, My original plan 
(when I was still considering using pymoo) was to assign failed trials 
artificial objective values like `+inf` and `-inf`. Once I decided not 
to depend on a package anymore, that stopped making sense. 

Luckily, the implementation that actually shipped ended up being much
simpler!

Failed or otherwise ineligible trials are removed from Pareto ranking
entirely and all receive one shared rank after every real Pareto tier.

That means:

- successful trials always outrank failed ones
- no fake objective values are invented
-  `TrialRecord` is never modified
- ranking only exists temporarily while breeding a generation

`TrialHistory` always remains the source of truth.

---

## Why I didn't use a prebuilt package

There are packages that already implement NSGA-II and the original plan was 
not to write it all myself. I ended up deciding against it for a few reasons:

First, its primary execution model assumes ownership of the evaluation loop.

This project doesn't have a cheapo objective function. Each candidate means
launching a real subprocess and waiting waiting waiting for a worker to 
finish. We already have the controller file which owns that lifecycle!

Second, using pymoo would require translating everything into and out of its
own internal representation, which is bunk because our project already has 
its own concepts of:

- `TrialRecord`
- `CandidateConfiguration`
- `OptimizationContract`
- objective directions
- execution status
- metrics validity

Every adapter between those types and pymoo would be another place for bugs
to hide.

Finally, the populations here are intentionally small. The default population is 
only a handful of candidates, and the expensive part of the run is model 
training, not non-dominated sorting.

At that scale, writing the ranking logic directly was just more simple than 
building and maintaining any sort of adapter layer.

---

# What actually shipped

The final implementation lives entirely in `search/nsga2.py`.

It implements:

- non-dominated ranking
- crowding distance
- binary tournament selection
- uniform crossover
- mutation
- duplicate detection
- generation management through `_pending_children`

Rather than implementing another version of Pareto dominance, it repeatedly
reuses the project's existing `pareto.build_pareto_front()` function to peel
off one Pareto layer at a time. 

Mutation also reuses `random_search._sample_value()` instead of introducing
another parameter sampler. Duplicate detection reuses
`random_search.candidate_key()`, and retries up to
`_MAX_DUPLICATE_ATTEMPTS = 100` before giving up on a candidate. It's the 
same limit `RandomSearch` already uses, kept consistent on purpose rather 
than picking a new number.

Crossover is uniform: each parameter independently has a 50/50 shot of
coming from either parent. That's the one crossover strategy that behaves
identically for FLOAT, INTEGER, and CATEGORICAL parameters so there's no
meaningful blend between two category names. This means we never even 
needed a parameter-kind-specific crossover rule, nice!

Population size and mutation rate aren't defined anywhere in the original
project spec, since NSGA-II itself is outside the MVP.

- `population_size = min(max(2 * num_parameters, 4), 10)` which is enough
  candidates per generation to actually rank and breed from, without one
  generation eating an entire small example runs trial budget.
- `mutation_rate = 1 / num_parameters` which targets one mutated parameter per child on average, This is a totally standard genetic-algorithm starting point.
- `_default_population_size()` also clamps to the finite search space size.
This was found via testing, not the original design. Without it, a tiny space (like
one categorical parameter with 2 values) would demand the default minimum
population of 4, and the first generation would fail instead of correctly
reporting `search_exhausted`.

The only existing project file that needing changing was `search/registry.py`, which
registers `"nsga2"` alongside `"random_search"`. Everything else is totally
additive and maintains the project's modular structure. :)

Files it actually leverages, confirmed by import:

- `pareto.py` -- `build_pareto_front()`, `is_eligible()`
- `search/random_search.py` -- `_sample_value()`, `candidate_key()`,
  `_finite_space_size()`
- `search/base.py` -- `ProposalResult`, the `SearchAlgorithm` protocol it
  implements
- `models.py` -- `CandidateConfiguration`, `OptimizationContract`
- `records.py` -- `TrialRecord`, read-only
- `numpy` -- `np.random.default_rng(seed)` for all randomness, same
  approach `RandomSearch` already uses

---

# Current limitations

Standard NSGA-II combines the parent and offspring populations, ranks the
combined population, and keeps only the best survivors, rather than letting
one completed generation breed the next directly.

```text
parents
     \
      +------+
             |
offspring ---+
             |
             ↓
non-dominated sorting
             ↓
best N survive
```

That elitist survival step is implemented: `NSGA2` tracks the surviving
parent population's trial IDs between calls to `propose()`, and each new
generation is bred from `_select_survivors()` over the combined
parent-plus-offspring pool rather than from the offspring alone.

The current implementation still keeps its pending child queue entirely in
memory. If the process crashes halfway through a generation, any children
still in the queue are lost. (That isn't unique to this algorithm, since the
project currently doesn't persist the internal state of random search
either.)

The ranking, selection, mutation, survivor-selection, and
generation-management logic is covered by `tests/test_nsga2.py`.

---

# Optional improvements

None of these are required for the current implementation to work. They're
just ideas that would make it more complete.

## Better finite-space handling

The implementation correctly detects complete search exhaustion.

One edge case still exists when fewer unused candidates remain than the
normal population size. Allowing the final generation to shrink instead of
trying to build a full batch would avoid an unnecessary
`proposal_failed`.

## Crash recovery

Adding persistent `_pending_children`, RNG state and current parent population
would make interrupted runs resumable without regenerating future generations 
from scratch.

I intentionally didn't solve that here because the rest of the project
doesn't solve it either.