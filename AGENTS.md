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

- Last updated: 2026-08-01
- Current state: `black_box_optimizer.controller` now provides
  `ApplicationController`/`ControllerState` per TDS §5, implementing the
  SELECTING/GATING/EXECUTING/RECORDING/EVALUATING loop against the real
  search/runner/records/persistence modules. FINALIZING is an intentional,
  explicit seam (`_finalize()` raises `NotImplementedError`) since neither
  the ParetoFront sweep nor reporting exist yet.
  `black_box_optimizer.pareto` now provides `is_eligible()` per TDS §8.1,
  implemented exactly to the literal spec pseudocode; the dominance
  comparison and ParetoFront sweep (§8.2/§8.3) remain unimplemented.
  `black_box_optimizer.persistence` now provides `RunDirectory`/
  `create_run_directory()`/`CheckpointError` per TDS §9, implementing the
  run/per-trial directory layout, `history.csv`'s schema, and atomic
  checkpointing (§9.1/§9.2/§9.4); `pareto_front.csv`, `summary.txt`, and
  `resolved_config.json` remain unimplemented pending the ParetoFront
  sweep and a JSON serializer that doesn't exist yet.
  `tests/integration/test_full_pipeline_real_worker.py` (merged
  separately, PR #8) upgraded the real-chain proof beyond the original
  one-trial slice to cover every `execution_status` `runner.py` can
  actually produce (`completed`/`launch_failed`/`timed_out`/
  `process_failed`) plus boundary parameter values. `KNOWN_ISSUES.md` is
  a new file tracking confirmed, reproduced bugs; its one open entry
  documents that `runner.py` captures a crashed worker's stdout/stderr
  but never returns it, so `process_failed` trials lose their error
  message and `persistence.py` can't yet preserve trial-local
  stdout/stderr per TDS §10.4. Added a `Persistence` row to TDS §3.1's
  component matrix and §11.2's dependency table (branch
  `work/2026-08-01-1330-tds-docs`) -- neither table had one despite
  `persistence.py` being a real, listed module.
  Immutable configuration and candidate models are in
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
  execution-result shape. `examples/iris_torch/worker.py` is now
  implemented per the demonstration contract: accepts learning_rate,
  hidden_size, epochs, batch_size, and returns validation_accuracy,
  validation_loss, training_time_seconds. `examples/` and
  `examples/iris_torch/` are now importable packages (added `__init__.py`
  files) so `tests/test_worker.py` can reference it directly.
  `examples/iris_torch/iris-data.csv` is the bundled dataset (converted
  from a course-provided space-separated file to a labeled CSV; no
  network access at runtime, satisfying the locality rule). This is the
  team's second dependency addition; `requirements.txt` now also
  includes `torch`. Controller and Pareto behavior do not exist yet.
  Aligned Draft v0.2 planning documents remain in
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
  `examples/iris_torch/worker.py`'s network architecture (one hidden
  layer, `Linear -> ReLU -> Linear`), `CrossEntropyLoss`, `SGD`, an 80/20
  train/validation split, and a fixed internal seed are all our own
  choices -- the demonstration contract only specifies the CLI/metrics
  interface, not the model internals.
  KNOWN CONTRACT DEVIATION: `ControllerState` omits TDS §5.1's literal
  `INITIALIZING` state -- `ApplicationController` accepts already-built
  configuration objects and an already-created `RunDirectory` rather
  than loading JSON or creating the run directory itself, so that setup
  happens before construction, not as a state transition. Checked
  against TDS §10.3's failure handling matrix directly: `RECORDING`
  catches `persistence.CheckpointError` and routes to
  `FAILED -> FINALIZING` with `fatal_error`, the one case where
  `RECORDING` (or `EXECUTING`, which never does) actually uses the
  `FAILED` transition §5.1 allows for it, since `launch_failed`/
  `process_failed`/`timed_out` are explicitly "continue if policy
  permits" rows in that same table, not fatal ones. Mid-worker
  `KeyboardInterrupt` cancellation (§5.3 and §10.3 both expect a
  cancelled trial to still be recorded) is not implemented; only
  pre-launch cancellation is handled for real, since safely terminating
  the child would mean changing `runner.py`'s blocking
  `subprocess.run()` call, a contract change needing team agreement
  first. `persistence.py`'s `metric.*` column ordering is sorted
  alphabetically for determinism, which TDS §9.2 doesn't actually
  mandate. Trial directories stay zero-indexed (`trial_0000`) to match
  `trial_id` everywhere else in the project, rather than the
  1-indexed-looking example in §9.1.
- Verification: All 129 tests pass under local Python (3.13.14 not
  available in this environment; verified under 3.14 instead). The
  repository hygiene checker passes without line-length advisories. A
  `pyflakes` pass across the whole project found zero issues in any file
  this work touched. The Iris JSON loads through the public loader, and
  all 60 pages of the four v0.2 planning documents were previously
  rendered and visually reviewed. `tests/integration/test_one_trial_slice.py`
  confirms real (non-mocked) cross-module execution: RandomSearch
  proposes a candidate, a metrics CSV is written to disk and read back
  through the real `metrics.py`, `build_trial_record()` builds a real
  `TrialRecord`, and it's appended to a real `TrialHistory`, gated by a
  real `StopPolicyEvaluator` at each step (its "worker" step is still a
  simulated CSV write, not `runner.execute()`, despite both now being
  mergeable -- see Next work). Separately, the full real chain (search,
  the real `runner.execute()`, the real `worker.py`, `metrics.py`, and
  `build_trial_record()`) was verified together outside the test suite:
  actual subprocesses training actual small neural networks on the real
  Iris data, producing plausible accuracy (86-97% in spot checks).
  `tests/integration/test_full_pipeline_real_worker.py` (PR #8) later
  replaced the simulated-worker approach above with a real chain
  covering every `execution_status` `runner.py` can produce. controller/
  pareto/persistence work: 177 tests passing across `test_controller.py`,
  `test_pareto.py`, `test_persistence.py`, and the real end-to-end
  `tests/integration/test_controller_real_pipeline.py` (proves the
  actual `ApplicationController` class orchestrates real search/runner/
  worker/persistence together, not just simulated pieces). Hygiene
  checker passes clean. Grepped the whole repository for AI/Claude/
  Anthropic references: none found.
- Next work: `controller.py`'s `FINALIZING` seam and `pareto.py`'s
  dominance/ParetoFront sweep (§8.2/§8.3) are the two large remaining
  pieces, and they're linked -- `_finalize()` can't do anything real
  until the sweep exists. `results.py` and `reporting.py` remain
  unclaimed and are also blocked on the same sweep. `persistence.py`'s
  `pareto_front.csv`/`summary.txt` are blocked the same way;
  `resolved_config.json` needs a JSON serializer that doesn't exist yet
  (`config_loader.py` only reads). Two real `runner.py` gaps are
  tracked in `KNOWN_ISSUES.md`: `process_failed` trials lose their
  error message, and stdout/stderr get captured then discarded --
  fixing that unblocks both the error-message issue and
  `persistence.py`'s trial-local `stdout.txt`/`stderr.txt` preservation
  (TDS §10.4). `cli.py` is blocked on the controller's `FINALIZING`
  producing real, exit-code-distinguishing outcomes. Follow change
  control only if implementation requires a documented contract or
  architecture change.
