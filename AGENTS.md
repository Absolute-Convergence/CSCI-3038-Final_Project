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
  noninteractive plot of the first two declared objectives. Hyperloop (the
  Tkinter/Pillow GUI) has landed on `main` per
  `docs/gui-handoff-next-steps.md`'s handoff plan. NSGA-II (`nsga2`) is now a
  second, opt-in `SearchAlgorithm`, registered alongside `random_search` in
  `ALGORITHM_REGISTRY` with no changes to `controller.py`, `pareto.py`,
  `persistence.py`, `history.py`, or `records.py`. It lives entirely in
  `black_box_optimizer/search/nsga2.py`, reusing
  `pareto.build_pareto_front()`/`is_eligible()` for non-dominated ranking and
  `random_search._sample_value()`/`candidate_key()`/`_finite_space_size()`
  for mutation, duplicate detection, and exhaustion; it depends on no
  external MOEA library (no pymoo). `config_loader.py`'s algorithm-name
  validation now checks `ALGORITHM_REGISTRY` instead of hardcoding
  `random_search`, so any registered algorithm is selectable from a config
  file. Design notes, including an after-the-build account of what actually
  shipped, live in `docs/decisions/2026-08-02-nsga2-search-algorithm-design.md`.
  A new synthetic benchmark, `examples/zdt1_benchmark/`, evaluates ZDT1
  (Zitzler-Deb-Thiele) directly -- two closed-form formulas, no model
  training, real subprocess per trial through the same `WorkerSpec`/
  `runner.py` contract every other worker uses. `compare_search_algorithms.py`
  in that directory scores `random_search` against `nsga2` by hypervolume
  relative to ZDT1's known-optimal Pareto front, across multiple seeds.
- Integration record: PR #20
  (`work/2026-08-01-2348-nsga2-search-algorithm`) merged NSGA-II, its design
  doc, the `config_loader.py` registry-based algorithm validation fix, a
  40-test unit suite for `nsga2.py`, and a real-worker end-to-end pipeline
  test into `main` as merge commit `7f5b0c4`. PR #21
  (`work/2026-08-03-1810-zdt1-benchmark`) merged the ZDT1 benchmark worker,
  the comparison script, and a 12-test suite for the ZDT1 formula and
  hypervolume math into `main` as merge commit `fc5121a`. PR #22
  (`work/2026-08-03-1903-test-bug-audit`) merged a coverage-closing pass
  (91% -> 99% overall, 80 new tests, 363 total) into `main` as merge commit
  `d72d3c9`. PR #23 (`work/2026-08-03-1940-bug-hunt`) fixed the
  order-dependent crowding-distance bug in `nsga2.py` into `main` as merge
  commit `341abdb`. PR #24 (`work/2026-08-03-2011-negative-seed-bug`)
  fixed the negative-seed CLI crash by rejecting it at `AlgorithmSpec`
  construction into `main` as merge commit `faae4d8` (see
  `KNOWN_ISSUES.md`'s Resolved section for both). All five working
  branches deleted locally and remotely per convention.
- Decisions: NSGA-II is hand-implemented rather than depending on pymoo --
  pymoo's primary API assumes ownership of the evaluation loop (batch
  `minimize()`), doesn't fit this project's one-worker-at-a-time controller,
  and would need its own objective-direction/domain-type translation layer;
  population size is intentionally small (`clamp(2 * num_parameters, 4,
  10)`), so pymoo's batch-vectorization advantage wouldn't pay off here
  anyway. Failed/ineligible trials get one shared worst rank rather than an
  invented objective value, and `TrialHistory` is never rewritten with a
  fake value. Population size also clamps to the finite search-space size
  when smaller than the usual floor -- found via testing, not the original
  design. The ZDT1 benchmark uses n=4 decision variables, not the original
  paper's n=30 -- measured directly (20,000 random samples) that n=30's 29
  nuisance parameters put the optimal region out of reach for either
  algorithm within any practical trial budget, given this project's
  uniform-reset mutation operator (resamples a fresh random value rather
  than nudging toward a better one). Generated comparison output (raw CSV,
  chart) is gitignored under `examples/zdt1_benchmark/comparison_results/`
  as regenerable evidence, same treatment as `/runs/`.
- Verification: All 366 tests pass under Python 3.13.14 (the course-required
  interpreter; `.venv` was rebuilt from a stray 3.14.6 install during the
  NSGA-II checkpoint and has stayed on 3.13.14 since). Overall package
  coverage is 99% (1385 statements / 452 branches; 20 of 22 source files at
  100%) after a dedicated coverage-closing pass, verified with `coverage.py`
  (a local dev install, not a project dependency) -- see the prior
  checkpoint's notes for the two files that stay short of 100% and why
  (`__main__.py`'s subprocess-tracing gap, `controller.py`'s two genuinely
  unreachable RECORDING guards). The hygiene checker passes across 60
  source files with the same non-blocking line-length advisories as before.
  A real 5-seed x 500-trial x 2-algorithm ZDT1 comparison (5,000 real
  subprocess trials) ran successfully: NSGA2 averaged 64% more hypervolume
  than random_search (0.389 vs 0.237, as a fraction of the true-optimal
  hypervolume: 44.4% vs 27.0%) and was more than twice as consistent run to
  run (stdev 0.069 vs 0.157).
- A separate, dedicated adversarial bug-hunt pass (six parallel reviewers,
  one per module cluster, run immediately after the coverage pass above
  reached 99% with no bugs found) found 11 real, independently-reproduced
  bugs and 4 test-quality gaps that coverage alone never surfaces. Two
  (the `nsga2.py` crowding-distance bug, the negative-seed CLI crash) are
  fixed as of PR #23/#24; the other 9 are still open. Full details, each
  with a concrete repro, are recorded in `KNOWN_ISSUES.md` (Open and
  Resolved sections) -- read that file before starting the next branch
  instead of re-deriving this list. One of the remaining 9 is still
  critical: a `KeyboardInterrupt` during report-writing mislabels a
  genuinely completed run as "cancelled" (fix is small -- add
  `FINALIZING` to `controller.py`'s `_UNSAFE_TO_CANCEL_STATES` -- but
  changes observable behavior of a controlled contract file, worth a
  quick confirm before landing it).
- Next work: Fix the remaining bugs in `KNOWN_ISSUES.md`'s Open section,
  starting with the KeyboardInterrupt/cancelled-mislabeling one, on a new
  `work/YYYY-MM-DD-HHmm-bug-hunt`-style branch. Once those land, older
  possible directions if picked back up: a harder real-worker benchmark
  (Iris's objectives are too correlated to exercise Pareto diversity),
  parallel trial execution (a genuine architecture change to
  `controller.py`'s one-worker-at-a-time loop, not yet approved), or
  NSGA-II elitism (combining parent and offspring populations before
  selection, the biggest remaining gap versus canonical NSGA-II per its
  design doc).
