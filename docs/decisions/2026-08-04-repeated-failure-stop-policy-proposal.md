# Repeated-Failure Stop Policy Proposal

- **Status:** proposal -- awaiting team review, not yet implemented
- **Author:** Emily Tew
- **Scope:** `StopPolicyEvaluator` (`black_box_optimizer/stop_policy.py`) and
  the controller's use of it

---

# The problem

`StopPolicyEvaluator` is explicitly scoped to one check. Its own docstring
says so: *"This evaluator is only responsible for the `maximum_trials`
termination reason."* A trial failing has no effect on whether the next one
gets launched -- the controller keeps proposing and running trials purely
based on `len(history) < max_trials`, regardless of how many of those
trials actually produced usable results.

That's fine when failures are incidental (a rare timeout, one bad candidate
near a numeric edge case). It's not fine when the failure is structural --
a JSON configuration that doesn't match what the worker actually expects.
That kind of mismatch is static for the whole run: it doesn't depend on
which candidate values got sampled, so it fails the same way on trial 1 as
it would on trial 150. Reproduced directly this session: an Iris config
missing two required worker parameters produced 8/8 identical
`process_failed` trials, each with the exact same error message, burning
the entire trial budget for zero usable output.

We can't detect this class of problem statically -- the optimizer treats
the worker as opaque and has (deliberately) no protocol for asking it what
arguments it expects ahead of time, and we explicitly do not want to
require worker cooperation (no `--describe` contract, no schema workers
must implement). But we don't need static detection. We just need to
notice, dynamically and language-agnostically, that failures aren't random
noise -- and stop before wasting the rest of the budget.

---

# The big idea

Don't try to understand *why* trials are failing. Just notice *that* the
failure rate is statistically inconsistent with ordinary flakiness, and
stop early with a new termination reason once it is.

Real workers can be a little flaky -- an occasional timeout, one candidate
that happens to produce a non-finite metric near a numeric boundary. So the
policy shouldn't trigger on any failure, or even on a short losing streak.
It should trigger when the observed failure rate would be very unlikely if
the worker were only failing at some acceptable baseline rate.

That's a one-sided hypothesis test, and it naturally does the thing this
proposal needs: how much evidence is required to conclude "this run is
broken" depends on how many trials have run so far, tightening as more
evidence accumulates.

---

# Two mechanisms considered

## A. Flat threshold with a minimum sample floor

Simplest option: don't evaluate anything until some minimum number of
trials have run (so one unlucky trial can't trigger it), then stop if the
failure fraction among trials-so-far exceeds a fixed percentage (say 70%).

Easy to explain, easy to configure, but the threshold is arbitrary and
doesn't adapt: whatever minimum sample size and percentage get picked, they
either overreact on legitimately noisy small samples or underreact once the
trial count is large enough that even 70% failing is overwhelming evidence
of a real problem.

## B. Binomial-tail significance test (recommended)

Pick a baseline acceptable failure rate `p0` (an assumed ceiling on how
often even a healthy worker fails from ordinary flakiness -- proposing
`p0 = 0.3`) and a significance level `alpha` (proposing `alpha = 0.01`, the
same discipline used for the ZDT1 benchmark's Mann-Whitney test). After
each trial, compute the exact one-sided binomial tail probability of
seeing at least this many failures out of this many trials if the true
failure rate really were `p0`:

```python
import math

def binomial_tail_prob(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact, no scipy dependency."""
    return sum(
        math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        for i in range(k, n + 1)
    )
```

If that probability drops below `alpha`, the observed failures are
statistically inconsistent with ordinary flakiness -- stop with a new
`"excessive_failures"` termination reason.

Verified by hand before writing this proposal (baseline `p0 = 0.3`):

| trials (n) | failures (k) | P(X >= k \| n, p0=0.3) | triggers at alpha=0.01? |
|---|---|---|---|
| 1 | 1 | 0.30000 | no |
| 3 | 3 | 0.02700 | no |
| 5 | 5 | 0.00243 | **yes** |
| 8 | 6 | 0.01129 | no (barely) |
| 8 | 8 | 0.00007 | **yes** |
| 10 | 7 | 0.01059 | no (barely) |
| 10 | 8 | 0.00159 | **yes** |
| 20 | 12 | 0.00514 | **yes** |

This is exactly the shape the problem calls for: a single failure never
triggers it (30% is well within normal noise for one sample), a short
losing streak needs to be a real streak (5/5 straight failures is the
earliest an all-failure run would stop), and a run with some real successes
mixed in tolerates a higher failure count before triggering (8/6 failures
doesn't trigger, but 8 straight does, immediately). The reproduced Iris
case (8/8 identical failures) would have stopped by trial 5 instead of
burning all 8 -- and would stop even earlier on a longer configured budget,
since the same 100%-failure streak crosses the threshold at n=5 regardless
of how large `max_trials` is.

---

# Open questions for the team

- **Is `p0 = 0.3` and `alpha = 0.01` reasonable, or too strict/lenient?**
  These aren't derived from anything -- they're a starting proposal, not a
  measured constant.
- **Should `p0`/`alpha` be configurable per project** (a new field under
  `stop_policy` in the JSON config), or fixed constants shared across every
  run? Configurability adds surface area; fixed constants might not fit
  every worker's real flakiness rate.
- **What counts as a "failure" here** -- any ineligible trial (matching
  `pareto.is_eligible()`, which already covers `process_failed`,
  `launch_failed`, `timed_out`, bad metrics, and a missing declared
  objective key), or only execution-level failures? Recommending the
  broader `is_eligible()` definition, since a worker that always runs but
  never produces usable metrics wastes the budget exactly the same way.
- **Does this belong in `StopPolicyEvaluator`, or should it be a separate
  collaborator the controller consults alongside it?**
  `StopPolicyEvaluator`'s docstring currently claims single responsibility
  for `maximum_trials` specifically; extending it changes that contract.
- **Naming**: `"excessive_failures"` vs. something else, and where it needs
  to be added (`TerminationReason` in both `stop_policy.py` and
  `records.py`/`results.py` wherever that literal type is duplicated).

---

# What this explicitly does not do

No worker cooperation of any kind -- no `--describe` flag, no schema a
worker must implement, no assumption about what language a worker is
written in or what its failure output looks like. The mechanism only ever
looks at whether a trial was eligible, never why. This is the tradeoff
called out during the discussion that led to this proposal: we cannot
statically validate "does this JSON config match this worker" without
requiring the worker to describe itself, so this proposal targets the
dynamic, worker-agnostic signal instead -- catching the *symptom*
(failures that don't depend on which candidate was tried) rather than the
*cause* (whatever's actually wrong with the config or worker).
