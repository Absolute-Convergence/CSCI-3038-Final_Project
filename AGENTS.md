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

## Documentation Authority

- Recorded human decisions and this repository guidance control current work.
- `docs/architecture-baseline.md` contains the approved architectural
  invariants and MVP boundaries.
- Draft v0.2 under `docs/planning_baseline_v02` is the current detailed design
  guidance but remains pending team ratification and technical approval. It is
  not silently amended by merged code or scratch notes.
- Draft v0.1 under `docs/planning_baseline_v01` is historical source material,
  not current implementation guidance.
- `README.md` reports implementation status on `main`; `KNOWN_ISSUES.md` tracks
  reproduced defects in merged code. Neither file changes a public contract.
- When merged code and a controlled contract disagree, preserve the behavior,
  record the conflict, and obtain human agreement before changing either side.

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

## Merged Interface Alignment Notes

These are descriptions of merged behavior, not permission to change a
controlled contract without human agreement:

- `load_configuration(path)` returns one immutable `ProjectConfiguration`.
- `create_algorithm(spec)` constructs the algorithm; the immutable
  `OptimizationContract` is supplied later to `propose(contract, history)`.
- `runner.execute(worker_spec, candidate, metrics_path)` returns private
  observation fields consumed immediately by the record factory.
- `build_trial_record(candidate, trial_id, metrics_path, execution_result)`
  constructs one immutable `TrialRecord`.
- `CandidateConfiguration` protects its mapping but does not prove legality
  against an `OptimizationContract`; that validation remains a required
  controller boundary before launch.
- The abbreviated TDS public-interface table differs from several of these
  signatures. Do not make code conform to that table, or treat these notes as
  an amendment to the TDS, without human agreement and document reconciliation.

## Scratch Memory - Update After Every Change

Keep this section brief and current. Update it after every repository change so
the next person can resume without reconstructing recent decisions.

- Last updated: 2026-08-01
- Current state: `main` now includes the partial `ApplicationController`, run
  directories and atomic history checkpoints, Pareto eligibility, and the
  previously implemented configuration, search, runner, records, history,
  stop-policy, and real Iris worker boundaries. Controller finalization, full
  Pareto evaluation, immutable results, reporting, and CLI composition remain
  incomplete. The active stack validates proposed candidates before launch and
  now implements eligibility, mixed-direction dominance, the complete
  history-ordered `ParetoFront`, and immutable `OptimizationResult` status
  semantics. Invalid proposals launch no worker and enter fatal finalization.
- Active ownership: Codex owns controller integration on
  `work/2026-08-01-1947-controller-repair`; Mel reviews the controller and
  result contracts. The approved initialization direction is recorded in
  `docs/decisions/2026-08-01-controller-integration.md`.
- Decisions: Mel authorized Codex to own the remaining controller integration,
  with Mel as controller and result-contract reviewer. Initialization is a
  separate application/composition concern rather than a lifecycle-controller
  state; this emergent direction from the draft TDS must be documented.
  Declared-domain validation is required before launch; domain-valid worker
  failures are recoverable trials, and exclusion grammar is outside the MVP.
  `CheckpointError`, partial valid fronts for cancelled/failed results, a
  separate Reporter, approved report defaults, `Popen` cancellation with a
  two-second grace period, and bounded 1,000-character last-line diagnostics
  are approved. Continue timestamped branches, checkpoint commits, non-squash
  merges, verified tags, and merged-branch cleanup.
- Verification: All 201 tests pass under Python 3.13.14. Python 3.13 has NumPy
  2.5.1 and CPU-only PyTorch 2.13.0 installed while unqualified `python` and
  `pip` remain on the intended AppData Python 3.11 installation. The hygiene
  checker passes across 37 source files with one line-length advisory in
  `tests/test_persistence.py`.
- Next work: Implement persistence final outputs and the separate Reporter,
  then replace the controller's finalization placeholder. Cancellation, bounded
  diagnostics, trial-local streams, CLI composition, and end-to-end verification
  follow.
