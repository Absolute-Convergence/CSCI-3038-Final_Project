# NSGA-II Search Algorithm: Design Notes

- Status: personal design notes for an in-progress exploration, not a team
  decision or a contract amendment
- Author: Emily
- Date: 2026-08-02
- Scope: an additional, opt-in `SearchAlgorithm` implementation, registered
  alongside `RandomSearch`. Purely additive -- no changes to `controller.py`,
  `pareto.py`, `persistence.py`, or any existing contract.

This version is heavily annotated on purpose. Every block below explains not
just *what* the design says, but *why* it says that, so this can double as a
"do I actually understand this" check before any code gets written.

## Why this and not just RandomSearch

The MVP boundary requires seeded `RandomSearch` -- the word is "required,"
not "exclusive." `ALGORITHM_REGISTRY` in `search/registry.py` was already
built as a plain dict mapping algorithm name to a factory function, which
only makes sense as a design if more than one algorithm was always meant to
be pluggable into it. Adding a second entry isn't fighting the architecture,
it's using the extension point that's already there.

Why NSGA-II specifically, and not some other algorithm: this project's whole
point is multi-objective optimization with a Pareto front, not a single
"best" score. NSGA-II (Non-dominated Sorting Genetic Algorithm II) is the
standard, textbook algorithm for exactly that problem shape. Random search
never uses what it learned from earlier trials -- every draw is independent.
NSGA-II's entire premise is the opposite: rank what you've already tried,
breed the next batch from the better half, and repeat, so results should
trend upward with each generation instead of just accumulating luck.

## The core architecture: two separate worlds, joined by a translator

```text
TrialHistory                 pymoo
    parameters      ----->   decision variables
    # our named, typed parameters (learning_rate=0.05, batch_size=16)
    # become a plain numeric vector pymoo can do math on

    OptimizationContract ->  problem definition
    # bounds, parameter kinds, and objective directions become pymoo's
    # own "Problem" description of the search space

    metrics          ----->  objective values
    # accuracy/loss/whatever becomes a plain numeric vector too --
    # pymoo has zero opinion about what the numbers *mean*
```

Why this separation matters: pymoo has never heard of a `TrialRecord`, a
subprocess, a worker script, or a CSV file, and it never needs to. It only
ever wants two things -- numbers describing a candidate, and numbers
describing how that candidate did. Every domain-specific thing this project
cares about (metrics validity, execution status, persistence, cancellation)
stays entirely outside pymoo's world. Our job is only to be the translator
standing at the boundary between the two. This is *good* -- it means a bug
in the translator is easy to isolate (it can only be about conversion, not
about genetics), and a bug in the genetics is easy to isolate too (pymoo's
own code is already tested by its own maintainers).

```text
Controller
    asks for one candidate at a time, same as it does for RandomSearch
    # the controller does not know or care that a "generation" concept
    # exists on the other side of this call -- from its point of view,
    # every SearchAlgorithm looks identical: propose(contract, history)

GeneticAlgorithm (our SearchAlgorithm implementation)
    holds a queue of already-generated, not-yet-proposed children
    # this is the ONE piece of real, private state this class needs --
    # see "Open problem #3" below for why this can't be avoided

    generates a new batch only when that queue is empty
    # i.e., "did I already hand out everyone from this generation?"

    reads completed TrialRecords from history to know when a generation
    is complete
    # this class never has to be *told* a generation finished -- it can
    # always figure that out itself just by counting history

Pymoo adapter (inside GeneticAlgorithm, or a small helper module)
    converts parameter dicts into decision vectors
    converts valid metrics into objective vectors
    converts failed/ineligible trials into penalty vectors
    # ^ this bullet is "Open problem #2" below -- do not skip it,
    #   it's the single easiest thing to get quietly wrong

    calls pymoo's ask/tell interface (not `minimize()` -- see below)
    # ^ this is "Open problem #1" -- the single biggest API-fit risk

TrialHistory
    stays completely unchanged; still records real outcomes only
    # nothing about adding this algorithm touches history.py, records.py,
    # persistence.py, or controller.py at all -- if you find yourself
    # wanting to edit one of those files for this work, stop and
    # reconsider, because it probably means the adapter design is wrong
```

## Where a completed generation comes from

```python
# every trial -- success OR failure -- still consumes exactly one
# population "slot," in the order it was proposed. that's what makes
# this slice trustworthy: it's not "the last N trials that happened to
# succeed," it's "the last N trials, period," which is exactly one full
# generation as long as generations are always proposed in fixed-size
# batches (which this design guarantees -- see the propose() sketch below)
generation = history[-population_size:]
```

Why this is safe to rely on: it would NOT be safe if failed trials were
silently skipped or excluded from the count, because then a generation of
"nominally 8" might actually only have 6 real slots reflected in history,
and the next slice boundary would land in the wrong place, mixing children
from two different generations together. The fix for that risk is the
penalty policy in Open problem #2 -- failed trials still occupy a slot,
they just occupy it with a deliberately terrible score.

## Open problem #1: pymoo normally wants to own the whole loop

```text
# the pattern most pymoo tutorials show:
population
    |
    v
evaluate EVERYONE at once (a vectorized function call)
    |
    v
population
    |
    v
evaluate EVERYONE at once again
    |
    v
... repeat until done, all inside one blocking call to minimize()
```

Why this doesn't fit us: "evaluate everyone at once" assumes you can hand
pymoo a batch of candidates and immediately get back a matching batch of
scores, as if the evaluation function were cheap and instantaneous. Ours is
not -- each candidate means launching a real subprocess, training a real
model, and waiting for it to finish, one at a time, because
`runner.py`/`controller.py` are built around "no more than one worker
process active at a time" and that isn't changing for this work.

**Resolution:** pymoo's ask/tell interface (`algorithm.ask()` /
`algorithm.tell()`) is built for exactly this situation -- an external loop
that controls timing, and only tells pymoo about results whenever it has
them. This still needs to be *verified directly against pymoo's real docs*
before committing further -- specifically, confirm it can produce offspring
from an already-evaluated population without pymoo trying to run its own
evaluation cycle internally. If ask/tell turns out not to support that
cleanly, the fallback is to use only pymoo's individual crossover/mutation/
selection operators as building blocks, and keep the whole generation loop
written by hand inside our own code. Either path works; ask/tell is just
less code to write ourselves if it holds up.

## Open problem #2: failed and ineligible trials (the important one)

```text
# WRONG mental model (this is what the simple diagram above implies,
# and it's not actually true):
TrialRecord.metrics  ALWAYS has valid numbers  ---->  objective vector

# ACTUAL reality in this project:
TrialRecord.execution_status can be:
    "completed"       -> probably has real metrics
    "process_failed"  -> no usable metrics at all
    "launch_failed"    -> no usable metrics at all
    "timed_out"        -> no usable metrics at all
TrialRecord.metrics_status can independently be:
    "valid"        -> numbers are real and finite
    "missing"      -> no metrics file was ever produced
    "malformed"    -> a metrics file existed but didn't parse
    "nonfinite"    -> a metrics file existed but had NaN/inf in it
```

Why this matters so much: this whole project's design (`metrics.py`,
`records.py`, `pareto.is_eligible()`) treats failure as a completely normal,
expected, first-class outcome -- not an edge case to bolt on later. A real
NSGA-II adapter has to have an explicit answer for "what objective value
does a failed trial get," or the very first run with even one crashed
worker will break in a confusing way (missing dictionary key, `None`
where a float was expected, etc.).

**Policy:** give every failed/ineligible trial a penalty value that's worse
than every valid trial could ever be, on every objective, respecting each
objective's own direction:

```python
# pseudocode, not real code -- direction-aware penalty assignment
def penalty_value(objective):
    if objective.direction is Direction.MINIMIZE:
        # for something we're trying to make SMALL (like loss),
        # the worst possible value is the largest possible value
        return float("inf")

    # objective.direction is Direction.MAXIMIZE
    # for something we're trying to make BIG (like accuracy),
    # the worst possible value is the smallest possible value
    return float("-inf")
```

```text
# stated plainly:
minimize loss:      failed trial's penalty value = +infinity  (worst)
maximize accuracy:  failed trial's penalty value = -infinity  (worst)
```

Why this specific approach (as opposed to just, say, zero): using an
actual infinity guarantees the failed individual is dominated by *every*
valid individual, no matter what scale the real objective values happen to
be on. A magic number like zero or -1 could accidentally NOT be the worst
possible value if the real objective's range includes something even worse
-- infinity can't have that problem.

**Caveat that needs verifying directly against pymoo, not assumed:** pymoo
typically converts every objective internally into a minimization problem,
negating maximized objectives rather than keeping our own
MAXIMIZE/MINIMIZE labels around. That means the adapter needs ONE
consistent convention -- most likely: always convert to pymoo's internal
minimization representation *first*, and only then decide the penalty sign,
rather than deciding the penalty sign using our own project-level direction
and hoping it survives pymoo's conversion unchanged. Mixing "our" directions
with "pymoo's" directions in the same function is exactly the kind of thing
that produces a bug that only shows up on the MAXIMIZE objectives, not the
MINIMIZE ones (or vice versa) -- annoying to debug because half the tests
would pass.

**This does not touch real history, ever -- worth repeating clearly:** the
penalty value is a fiction that exists *only* inside the pymoo adapter's
translation step, for the duration of one call, and disappears immediately
after. `TrialHistory` is never rewritten to pretend a failed trial produced
infinity. Its real `execution_status`, `metrics_status`, and
`error_message` stay exactly what they actually were, forever. If a test
ever accidentally checks a real `TrialRecord`'s metrics and finds infinity
in it, that's a sign the adapter leaked its internal fiction somewhere it
shouldn't have.

**Alternative that was considered and rejected:** exclude failed trials
from parent selection entirely, instead of penalizing them. Rejected
because it shrinks the effective population size unpredictably (a
generation with 3 failures out of 8 only has 5 real candidates to select
parents from), which then needs its own fallback logic for "what if too
few valid parents remain." Penalizing keeps the population a fixed,
predictable shape every single generation, which is simpler to write tests
against.

## Open problem #3: this can't honestly be fully stateless

```text
# the tempting-but-wrong idea:
# "just re-derive everything from history every single propose() call,
#  and keep zero internal state at all"
#
# why it doesn't actually work:
propose() call #1 (first child of generation 2 needed)
    -> re-run evolution from scratch on generation 1's results
    -> get back a FRESH random batch of 8 children
    -> hand out child at position 0

propose() call #2 (second child of generation 2 needed)
    -> re-run evolution from scratch on generation 1's results AGAIN
    -> unless this is seeded with painstaking, deliberate care to
       reproduce the EXACT SAME random draws as call #1, this produces
       a DIFFERENT batch of 8 children, not the same one
    -> "position 1" of this new batch has no relationship to
       "position 1" of the batch from call #1
    -> now you've silently thrown away child 0 from call #1 and are
       handing out an unrelated stranger instead
```

Why the naive "no state at all" idea breaks: asking pymoo to evolve a
generation is a single call that produces the *whole* next generation's
children at once, and that process involves randomness (which parents get
picked, how crossover mixes their genes, whether/how mutation perturbs the
result). `propose()`, on the other hand, only ever gets to return ONE
candidate per call. Something has to bridge that gap -- either genuinely
reproducible, deliberately-reseeded-identically-every-time randomness
(technically possible, but fragile and easy to get subtly wrong), or just...
remembering the batch you already made.

**Resolution:** keep one small, explicit, honest piece of real state --
a queue holding this generation's already-generated children, computed
once per generation boundary, handed out one at a time as `propose()`
gets called:

```python
# pseudocode sketch, not real code
class GeneticAlgorithm:
    def __init__(self, ...):
        # this queue is the ONLY piece of state beyond what RandomSearch
        # already needs (a seeded RNG). everything else gets re-derived
        # from history fresh, every call.
        self.pending_children: deque[CandidateConfiguration] = deque()
        self.generation_number = 0

    def propose(self, contract, history):
        if self.pending_children:
            # still have leftover children from the last time we bred
            # a generation -- just hand out the next one, no new work
            return candidate_result(self.pending_children.popleft())

        if not_enough_history_for_a_full_generation(history):
            # generation zero: no parents exist yet, so seed randomly,
            # same idea as RandomSearch, just with a fixed batch size
            return random_initial_candidate(contract)

        # otherwise: a full generation's results just landed in history,
        # and the queue is empty -- time to breed the next one
        evaluated_population = build_pymoo_population(
            history[-population_size:], contract,
        )  # <- this is where Open problem #2's penalty policy applies
        children = ask_pymoo_for_next_generation(evaluated_population)
        self.pending_children.extend(children)
        self.generation_number += 1

        return candidate_result(self.pending_children.popleft())
```

```text
# so the accurate, honest split of responsibility is:

TrialHistory   = durable truth
                 (what was evaluated, what succeeded or failed,
                  the real objective values, when a generation is done)

pending queue  = temporary, in-memory-only control state
                 (which children have already been generated,
                  which one comes back next)
```

This is a meaningfully more honest description than "TrialHistory is the
only source of truth and nothing else needs state" -- that phrase sounds
clean but isn't quite accurate. The correct claim is narrower: history is
the only source of truth about *what actually happened*, but the algorithm
still needs a small amount of its own bookkeeping about *what it has
already decided to do next but hasn't told anyone about yet.*

## Open problem #4: what if the process crashes mid-generation

```text
# scenario: 50 children get generated and queued, but only 12 are
# evaluated (their results are in history) before the process dies.
# the other 38, still sitting in pending_children, only ever existed
# in memory -- they're gone.
#
# three theoretical options:
#   1. accept it: a restarted run just regenerates a different
#      remainder of the generation from scratch
#   2. persist the pending queue itself into the run's checkpoint
#   3. reconstruct the exact same queue deterministically from the
#      prior generation + generation number + seed + operator config
```

**This is already answered by existing project precedent -- it is not a
new decision this design needs to make.** The MVP explicitly and
deliberately excludes "resume-after-interruption behavior" as a project-
wide boundary. Nothing in this codebase persists any search algorithm's
state across a process restart today -- not even `RandomSearch`, which
recomputes its own `attempted_keys` fresh from history on every single
`propose()` call, and would produce a different remainder of its own
sampling sequence on a fresh process too, given the exact same seed and
history. So option 1 (accept it) isn't a compromise being made just for
this algorithm -- it's the same behavior every existing algorithm in this
project already has. An in-memory-only `pending_children` queue needs no
special persistence work, because nothing else gets that treatment either.

## Build order (repeating it here so it lives with the design, not just
## in conversation)

1. Random generation-zero seeding only. No genetics yet.
2. Generation-boundary detection only, still no genetics -- prove the
   code can look at history and correctly say "generation N is complete."
3. Non-dominated sorting as a standalone function, tested against a small
   hand-worked fixture where the correct ranking is already known by hand.
4. Crowding distance as a standalone function, same treatment.
5. Selection (tournament, based on rank + crowding) as a standalone
   function, tested with an already-ranked fixture, no randomness needed.
6. Crossover and mutation as standalone functions last -- reuse
   `_sample_value()` from `random_search.py` for mutation instead of
   writing a new per-kind sampler from scratch.
7. Only now, wire everything above together inside `propose()`.
8. Register it: one factory function, one line in `ALGORITHM_REGISTRY`.

Verify pymoo's actual ask/tell API directly, for real, before committing
further to any of the above -- this entire adapter design assumes it exists
and behaves as described, and that assumption has not been confirmed
against pymoo's real documentation yet.
