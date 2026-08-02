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
- Current state: The integrated MVP includes configuration,
  declared-domain candidate validation, search, synchronous cancellable runner,
  immutable records/results, append-only history, atomic persistence, the full
  controller lifecycle, Reporter outputs, separate application composition,
  the module CLI, and the real Iris worker. It builds the complete
  history-ordered `ParetoFront`. Both result types live in
  `black_box_optimizer.results`; Pareto algorithms remain in
  `black_box_optimizer.pareto`. Invalid proposals launch no worker and enter
  fatal finalization. The separate `Reporter` writes the resolved JSON shape,
  deterministic Pareto CSV, text summary, and a
  noninteractive plot of the first two declared objectives. Controller
  finalization now builds and returns the authoritative result, delegates it to
  Reporter exactly once, and reaches `STOPPED` after successful output. The
  runner now uses synchronous `Popen`, terminates then kills after a two-second
  grace period when required, returns bounded last-line diagnostics, and makes
  complete decoded streams available for trial-local persistence. A cancelled
  worker produces exactly one cancelled record and a cancelled result. The
  separate application layer composes initialization and writes resolved config
  before trial 1; `python -m black_box_optimizer` is now runnable. The example
  Iris configuration explicitly launches its worker through `py -3.13`.
  Repository-root `runs/` contains generated local evidence and is ignored by
  Git. `docs/gui-handoff-next-steps.md` now records the recommended
  contract-preserving handoff from Charles's Tkinter/Pillow GUI to the existing
  CLI and run artifacts.
- Integration record: Mel reviewed and merged PRs #12 through #15. Because
  PRs #13 through #15 retained their stacked branch bases, a final
  main-targeted integration merge was required to land their already-reviewed
  trees on `main`. The approved initialization direction is recorded in
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
  merges, verified tags, and merged-branch cleanup. Hyperloop is the canonical
  product name; `black_box_optimizer` remains the implemented Python module
  and CLI namespace unless Mel approves a compatibility change.
- Verification: All 229 tests pass under Python 3.13.14. Python 3.13 has NumPy
  2.5.1 and CPU-only PyTorch 2.13.0 installed while unqualified `python` and
  `pip` remain on the intended AppData Python 3.11 installation. The hygiene
  checker passes across 51 source files without advisories. Matplotlib 3.11.1
  is installed for the approved Reporter plot work, and `pip check` reports no
  broken requirements. Git ignore verification covers the populated default
  `runs/` output directory. The 229-test suite and hygiene checker were rerun
  after adding the GUI handoff guide; both pass, and its referenced local paths
  exist.
- Next work: Charles should review and implement the contract-preserving GUI
  sequence in `docs/gui-handoff-next-steps.md`, beginning with complete
  multi-objective JSON construction, launching Hyperloop instead of the opaque
  worker, and loading the existing PNG/CSV/JSON artifacts. Do not add pickle or
  another reporting/CLI contract without Mel's prior approval. No other core
  MVP implementation work remains after the verified integration checkpoint.
