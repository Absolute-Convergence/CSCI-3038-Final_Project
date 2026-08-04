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

- Last updated: 2026-08-03
- Current state: Package preparation is active on
  `work/2026-08-03-1923-package-release-audit`. The branch integrates
  `origin/main` through `f9c9e64` and the local package checkpoint
  `beaefa7`. `pyproject.toml` targets distribution
  `hyperloop-optimizer` version 0.1.1. The approved installed commands are
  `hyperloop-optimizer` and `hyperloop-synthetic-worker`; the existing
  module CLI remains available. The wheel contains the optimizer core and the
  dependency-free synthetic worker, while the Iris worker, GUI, tests, and
  class-project material remain outside the installed package.
- Search-efficacy benchmark: The ZDT1 comparison harness now accepts
  `--seeds`, `--trials-per-run`, and `--output-dir`. Its default is 10 seeds x
  500 trials x 2 algorithms, or 10,000 real sequential worker subprocesses.
  The documented practical tiers are 1,000, 5,000, and 10,000 total trials,
  selected with 1, 5, or 10 seeds respectively. Do not use 20K/50K as package
  release gates; the verified 10K run already takes about 11.5 minutes.
- Cleanup: Checkpoint `beaefa7` removes 515 generated
  `optimizer_runs/` artifacts from Git tracking while preserving the local
  files under an ignored root directory. A repeated-run integration test
  verifies each Pareto report is derived only from its own authoritative
  result.
- Upstream integration: PR #22's coverage work, PR #23's deterministic NSGA-II
  tied-objective crowding-distance fix, and PR #24's negative-seed validation
  fix are integrated. Mel approved the remaining package-remediation contract
  choices. The ten remaining defects are fixed with focused regression tests;
  `KNOWN_ISSUES.md` has no confirmed open defect on this branch. The approved
  decisions are recorded in
  `docs/decisions/2026-08-03-package-release-bug-remediation.md`.
- Decisions: Publish one first PyPI release as v0.1.1 rather than reusing the
  existing v0.1.0 Git tag. Core direct dependencies are NumPy and Matplotlib;
  PyTorch stays exclusive to the Iris example, and Pillow may be installed
  only transitively through Matplotlib. Mel approved the dedicated
  `hyperloop-synthetic-worker` entry point so installed configurations do not
  guess the owning Python executable.
- Verification: After the package-remediation pass, all 394 tests pass on
  Windows under Python 3.13.14. The hygiene check passes across 62 source
  files. Fresh wheel and sdist builds pass `twine check`; the 30-entry wheel
  has only NumPy and Matplotlib as direct requirements and contains no GUI,
  Iris, tests, examples, documentation, or PyTorch. A fresh install outside
  the checkout passes both command help checks, `pip check`, and a four-trial
  synthetic-worker run with every required artifact. In that PyTorch-free
  environment, 380 tests pass and four Iris modules skip as intended. The full
  post-remediation ZDT1 benchmark completed 10,000 real subprocess trials in
  687.6 seconds. NSGA-II mean hypervolume was 0.3884 (44.3% of optimum,
  stdev 0.0924) versus Random Search 0.2113 (24.1%, stdev 0.1119), about 84%
  more dominated hypervolume. The ignored raw CSV contains 5,000 rows per
  algorithm and the ignored convergence PNG was regenerated.
- Cross-platform gate: The package workflow now builds and validates one wheel
  and source archive, uploads that exact distribution bundle, and requires the
  Windows, Ubuntu, and macOS jobs to install the uploaded wheel before running
  the core-compatible suite, dependency check, hygiene check, and installed
  synthetic-worker smoke test. Hosted-runner results remain pending until the
  branch is pushed and its pull request jobs complete. After this workflow
  change, all 394 tests, the 62-file hygiene check, a fresh wheel/sdist build,
  `twine check`, and the workflow's exact-wheel selection command pass locally.
- Next work: Push this branch, open its pull request, and require the package,
  Windows, Ubuntu, and macOS jobs to pass. After hosted verification, record
  the support claim and build the final archives from the exact release commit.
  The remaining release gates are recorded in
  `docs/package-release-checklist.md`.
