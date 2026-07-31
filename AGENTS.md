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
  Iris JSON is a loader-tested example. Search, execution, trial records,
  history, and Pareto behavior do not exist on `main` yet. Aligned Draft v0.2
  planning documents remain in `docs/planning_baseline_v02`; Draft v0.1 is
  preserved in `docs/planning_baseline_v01`.
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
- Verification: All 26 tests pass under Python 3.13 in 0.035 seconds.
  The repository hygiene checker passes without line-length advisories. 
  The Iris JSON loads through the public loader, and all 60 pages of the 
  four v0.2 planning documents were previously rendered and visually reviewed.
- Next work: Most of the one-trial verticle slice is complete. The only things
  that appear to be left is the controller. Eventually we will need to implement
  the search features as that remains one of the largest independent systems yet
  completed.
