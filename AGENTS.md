# Black Box Optimizer Repository Guidance

## Controlling Baseline

- Build a local Python 3.13.14 application that optimizes an opaque external
  worker through command-line parameters and a one-row CSV metrics file.
- Run synchronously and sequentially, with no more than one worker process
  active at a time.
- Create one immutable `TrialRecord` for every attempted execution and append
  it to an append-only `TrialHistory`.
- Evaluate eligible records with mixed-direction Pareto dominance and return
  the complete non-dominated `ParetoFront`.
- Do not import or inspect worker internals, invent a weighted winner, or add
  networking, concurrency, databases, resume behavior, or extra algorithms to
  the MVP.

The architectural baseline given by Mel controls when draft planning documents
conflict with current decisions. Record and explain conflicts before changing
public contracts or architecture.

## Ownership and Boundaries

- Mel owns the architectural baseline, internal contracts, optimizer/search,
  `TrialHistory`, persistence, Pareto evaluation, verification, and overall
  integration.
- Charles owns only the GUI. The GUI is an optional presentation layer and must
  not own or be required by optimizer logic.
- Reporting and result exports belong to their assigned teammate and must
  consume authoritative optimizer results without changing them.
- Example workers live outside the optimizer package. PyTorch may be used by
  the Iris example worker but must not become an optimizer dependency.

## Contract and State Rules

- Configuration, candidates, completed trial records, Pareto fronts, and final
  results must be immutable through public interfaces.
- `TrialHistory` may append internally but exposes only read-only tuple
  snapshots to search and evaluation code.
- Objective eligibility and Pareto membership are derived from a
  `TrialRecord` and `OptimizationContract`; do not store redundant flags.
- Preserve parameter order when constructing worker CLI arguments.
- Invoke workers with an argument list and `shell=False`.
- Give every authorized attempt a unique trial identifier and metrics path.
- Process success, metrics validity, and Pareto eligibility are distinct facts.

## Global Monolith Hygiene

- Run `python tools/check_monoliths.py` as part of verification.
- No source file may exceed 1,000 physical lines without explicit human
  approval recorded in `source_hygiene.json`.
- An approved exception must name the exact file, human approver, approval
  date, the file's single coherent responsibility, and why splitting it would
  make the design worse.
- Agents must not silently create, expand, or approve an oversized file.
- Lines should generally remain near 80 characters where practical. The
  checker reports long lines as design guidance rather than a hard failure.
- Tests, generated files, vendored dependencies, lockfiles, data files, and
  machine-generated artifacts may be excluded only through explicit checker
  configuration.
- The 1,000-line threshold is a warning signal for mixed responsibilities,
  coupling, and spaghetti-code risk. It is not a target or a score to game.
- Do not split code arbitrarily to satisfy the threshold. Split only at real
  responsibility boundaries with understandable interfaces.
- A collection of small but tightly coupled files can still violate this rule;
  passing the checker does not replace architectural review.
- The checker must clearly report unapproved oversized files and fail
  verification until a valid approved exception exists.

## Verification Expectations

- Add focused unit tests for pure validation and algorithms.
- Add integration tests across declared boundaries, beginning with the
  one-trial candidate-to-record vertical slice.
- Include a hand-calculated mixed-direction Pareto fixture.
- Keep tests deterministic and free of networking or concurrent worker trials.
- Before handing off a change, run the relevant tests and the global monolith
  checker, then update the scratch memory below.

## Change Control

Explain and obtain human agreement before changing JSON, CLI, CSV,
`TrialRecord`, eligibility, dominance, result, or persistence contracts. The
same requirement applies before adding a dependency or an excluded MVP
capability.

## Scratch Memory - Update After Every Change

Keep this section brief and current. Update it after every repository change so
the next person can resume without reconstructing recent decisions.

- Last updated: 2026-07-31
- Current state: Immutable configuration and candidate models are in
  `black_box_optimizer.models`. `black_box_optimizer.config_loader` now loads
  the approved JSON shape into an immutable `ProjectConfiguration`, preserves
  parameter and objective order, rejects malformed or contract-drifting input,
  and resolves relative worker paths from the configuration directory. The
  Iris JSON is a loader-tested example. `black_box_optimizer.metrics` now
  provides `read_trial_metrics()`, a one-row CSV metrics parser satisfying
  the Metrics Validity contract in the Technical Design Specification (§6.2,
  §2.3). `black_box_optimizer.records` now provides `TrialRecord` and
  `build_trial_record()` per TDS §6.3. `black_box_optimizer.stop_policy` now
  provides `StopDecision` and `StopPolicyEvaluator` (`before_trial()`,
  `after_trial()`) per TDS §5.3 -- the only TerminationReason it evaluates is
  `maximum_trials`; the other three are the controller's responsibility.
  `black_box_optimizer.history` now provides `TrialHistory`
  (`append()`/`snapshot()`), an in-memory, mutable-internally,
  read-only-externally ordered collection rejecting duplicate or
  out-of-order trial IDs; it does not perform any file I/O (that remains
  `persistence.py`'s job). `black_box_optimizer.search` (a new subpackage)
  now provides `ProposalResult`/`SearchAlgorithm` (`base.py`),
  `ALGORITHM_REGISTRY`/`create_algorithm()` (`registry.py`), and the seeded
  `RandomSearch` algorithm (`random_search.py`) per TDS §7. This is the
  team's first NumPy dependency; `requirements.txt` now exists and must be
  installed before running the test suite. `tests/integration/` (planned
  but previously empty) now has its first test,
  `test_one_trial_slice.py`, chaining metrics, records, history, search,
  and stop_policy together for real per the "Integration" row of TDS
  §12.2's minimum test layers table. `black_box_optimizer.runner` now
  provides `execute()`, the synchronous subprocess boundary, merged
  separately and confirmed compatible with `records.py`'s expected
  execution-result shape. Controller and Pareto behavior do not exist
  yet. Aligned Draft v0.2 planning documents remain in
  `docs/planning_baseline_v02`; Draft v0.1 is preserved in
  `docs/planning_baseline_v01`.
- Decisions: Source files over 1,000 physical lines fail verification unless an
  exact, documented human-approved exception exists. Near-80-character lines
  are advisory. Parameter JSON uses `kind`; `AlgorithmSpec` stores `name` and
  `seed`; candidate mappings are defensively copied into read-only views.
  Relative worker paths resolve from the configuration file's directory. Daily
  work uses `work/YYYY-MM-DD-HHmm-short-topic` branches, frequent checkpoint
  commits, and non-squash merges to `main`. After verification, an annotated
  `checkpoint/main-<topic>-YYYY-MM-DD-HHmm` tag preserves the merge checkpoint,
  and the completed local and remote work branches are deleted. The v0.2
  planning set is current guidance but remains pending team ratification and
  technical approval; each alignment change includes its reason. The loader
  treats the documented JSON fields as exact, rejects duplicate object keys
  and unsupported algorithms, and accepts only `random_search` for the MVP.
  `metrics.py` raises `MetricsFormatError` for structural CSV problems and
  `NonFiniteMetricError` for NaN/infinite values (both subclass `ValueError`)
  so `records.py` can map failures onto the `MetricsStatus` values
  (`missing`/`malformed`/`nonfinite`) from TDS §6.3 without string-matching
  exception messages. KNOWN CONTRACT DEVIATION: TDS §11.1 lists
  `build_trial_record(candidate, metrics_path, observed_fields)`, but the
  implemented signature is `build_trial_record(candidate, trial_id,
  metrics_path, execution_result)`. This follows the real `runner.py`
  (branch `work/2026-07-29-1918-runner`), whose `execute()` returns a dict
  with only `runtime_seconds`, `exit_code`, `timed_out`, `execution_status`,
  and `error_message` -- it does not bundle `trial_id` or `metrics_path`.
  Do not revert `records.py` to match the abbreviated §11.1 table without
  first confirming `runner.py`'s actual return shape (now merged and
  confirmed to match). `StopDecision` requires `continue_execution` and
  `termination_reason` to agree with each other (a reason is required
  when stopping, forbidden when continuing) since an inconsistent
  decision would be a silent bug in the controller loop.
  KNOWN CONTRACT DEVIATION: `registry.py`'s `create_algorithm(spec)` follows
  TDS §7.1's literal `Callable[[AlgorithmSpec], SearchAlgorithm]` factory
  type, not §11.1's abbreviated `AlgorithmRegistry.create(spec, contract)`
  summary -- a search algorithm's contract is passed separately to
  `propose()` and isn't needed at construction time. The
  duplicate-candidate retry limit in `random_search.py`
  (`_MAX_DUPLICATE_ATTEMPTS = 100`) is our own choice; TDS §7.2 requires a
  bounded limit but does not specify a number. `_sample_value()`'s
  categorical branch must use `np.asarray(choices, dtype=object)` before
  calling `Generator.choice()` -- without it, NumPy silently coerces mixed
  int/str choices into one common type (an int like `1` can come back as
  the string `"1"`); this was caught with a real reproduction, not
  theoretically. `RandomSearch.__init__` explicitly rejects negative seeds
  itself rather than letting NumPy's internal error surface, matching this
  project's convention of raising its own error messages.
- Verification: All tests pass under local Python (3.13.14 not available
  in this environment; verified under 3.14 instead) -- 118 from the
  search/integration work plus runner.py's own tests. The repository
  hygiene checker passes without line-length advisories. The Iris JSON
  loads through the public loader, and all 60 pages of the four v0.2
  planning documents were previously rendered and visually reviewed.
  `tests/integration/test_one_trial_slice.py` confirms real (non-mocked)
  cross-module execution: RandomSearch proposes a candidate, a metrics CSV
  is written to disk and read back through the real `metrics.py`,
  `build_trial_record()` builds a real `TrialRecord`, and it's appended to
  a real `TrialHistory`, gated by a real `StopPolicyEvaluator` at each
  step. This was additionally verified against the real, merged
  `runner.py` directly (not simulated): a real subprocess running a
  throwaway worker script, through `RandomSearch`, the real
  `runner.execute()`, `metrics.py`, and `build_trial_record()`, end to
  end.
- Next work: `runner.py` is now merged. `records.py`, `stop_policy.py`,
  `history.py`, and `search/` are all available and confirmed compatible
  with the real `runner.py`. `tests/integration/test_one_trial_slice.py`
  should be upgraded to call the real `runner.execute()` instead of its
  current simulated worker step, now that it's mergeable. The two large
  remaining pieces are the controller state machine (TDS §5) and
  `pareto.py` (TDS §8) -- both are fully specified with working
  pseudocode. `persistence.py`, `results.py`, `reporting.py`, `cli.py`,
  and the Iris worker script remain unclaimed. Follow change control only
  if implementation requires a documented contract or architecture
  change.
