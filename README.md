# Black Box Optimizer

CSCI 3038 final project by The Snek People.

Black Box Optimizer is a planned local Python 3.13.14 application for tuning an
external worker program without importing or inspecting that worker's internal
code. It will send candidate parameters through command-line flags, receive one
row of numerical metrics through a trial-specific CSV file, and return the
complete non-dominated Pareto Front across multiple objectives.

The repository is currently at the foundation stage. Immutable configuration
and candidate models exist, but search, worker execution, trial history, Pareto
evaluation, reporting, and the optional GUI have not been implemented.

See [docs/architecture-baseline.md](docs/architecture-baseline.md) for the
controlling foundation contracts and MVP boundaries.

The current planning guidance is the aligned Draft v0.2 document set in
[`docs/planning_baseline_v02`](docs/planning_baseline_v02). It remains pending
team ratification and technical approval. Draft v0.1 is preserved unchanged in
[`docs/planning_baseline_v01`](docs/planning_baseline_v01) as the historical
source baseline.

## Planned Runtime Flow

```text
JSON project configuration
          |
          v
Configuration loader -> immutable ProjectConfiguration
          |
          v
Seeded RandomSearch -> immutable CandidateConfiguration
          |
          v
Controller + StopPolicy -> synchronous local Runner
          |                         |
          |                         v
          |                external opaque worker
          |                         |
          |                         v
          |                 one-row metrics CSV
          |                         |
          v                         v
TrialRecord factory <------ execution and metrics observations
          |
          v
append-only TrialHistory
          |
          v
mixed-direction Pareto evaluator
          |
          v
OptimizationResult containing the complete Pareto Front
          |
          +----> reporting and exports
          +----> optional GUI presentation
```

Only the controller may authorize a worker launch. At most one worker process
will run at a time. Reporting and GUI code will consume optimizer results but
must not select candidates, change history, or replace the Pareto Front with a
weighted winner.

## Rough Planned File Structure

Items marked `implemented` exist now. Other modules show the intended
responsibility boundaries and will be added in later, approved changes.

```text
CSCI-3038-Final_Project/
|-- AGENTS.md                         # repository rules and scratch memory
|-- README.md                         # project overview and development map
|-- source_hygiene.json               # global source-file hygiene settings
|-- black_box_optimizer/
|   |-- __init__.py                   # implemented public model exports
|   |-- __main__.py                   # planned module entry point
|   |-- cli.py                        # planned CLI composition
|   |-- models.py                     # implemented immutable foundation types
|   |-- config_loader.py              # planned JSON parsing and validation
|   |-- controller.py                 # planned sequential lifecycle governor
|   |-- runner.py                     # planned synchronous subprocess boundary
|   |-- metrics.py                    # implemented one-row CSV parser
|   |-- records.py                    # planned TrialRecord construction
|   |-- history.py                    # planned append-only TrialHistory
|   |-- persistence.py                # planned durable history snapshots
|   |-- stop_policy.py                # planned maximum-trial decisions
|   |-- pareto.py                     # planned eligibility and dominance
|   |-- results.py                    # planned ParetoFront/OptimizationResult
|   |-- reporting.py                  # planned result export boundary
|   `-- search/
|       |-- base.py                   # planned search protocol/results
|       |-- registry.py               # planned built-in algorithm registry
|       `-- random_search.py          # planned seeded RandomSearch only
|-- examples/
|   `-- iris_torch/
|       |-- iris_config.json          # implemented example configuration
|       `-- worker.py                 # planned external PyTorch worker
|-- tests/
|   |-- test_models.py                # implemented foundation-model tests
|   |-- test_metrics.py               # implemented metrics-parser tests
|   |-- test_check_monoliths.py       # implemented hygiene-checker tests
|   |-- unit/                         # planned focused unit tests
|   |-- integration/                  # planned component-boundary tests
|   `-- fixtures/                     # planned workers, CSVs, and Pareto cases
|-- docs/
|   |-- architecture-baseline.md      # implemented controlling baseline
|   |-- planning_baseline_v01/        # preserved historical planning baseline
|   `-- planning_baseline_v02/        # current aligned draft planning guidance
`-- tools/
    `-- check_monoliths.py            # implemented source hygiene checker
```

The example worker remains outside `black_box_optimizer`. PyTorch may be a
dependency of that example, but it must not become a dependency of the
optimizer package.

## Component Responsibilities

| Component | Owns | Must not own |
| --- | --- | --- |
| Configuration loader | JSON parsing and complete pre-run validation | Worker execution or candidate search |
| Search algorithm | Legal untried candidate proposals | Subprocesses, stopping, or Pareto ranking |
| Controller | Sequential lifecycle and launch authorization | Worker internals or metric weighting |
| Stop policy | Maximum-trial decision | Candidate selection or process control |
| Runner | CLI construction, timeout, and process observations | Search or objective interpretation |
| Record factory | Metrics parsing and immutable trial evidence | History mutation beyond one append request |
| Trial history | Ordered append-only records and tuple snapshots | Ranking, deletion, or rewriting evidence |
| Pareto evaluator | Eligibility, mixed-direction dominance, full front | Weighted scoring or worker execution |
| Reporting/GUI | Presentation of authoritative results | Optimizer state or universal-winner selection |

The intended dependency direction is inward toward immutable contracts. The
Runner will not import search or Pareto code; RandomSearch will not import the
Runner; and the optimizer package will never import the example worker.

## Foundation Types Available Now

The `black_box_optimizer` package currently exports:

- `ParameterKind`
- `ParameterDefinition`
- `Direction`
- `Objective`
- `OptimizationContract`
- `WorkerSpec`
- `AlgorithmSpec`
- `StopPolicy`
- `ProjectConfiguration`
- `CandidateConfiguration`

These are frozen dataclasses or string enums. Ordered collections use tuples,
and candidate mappings are defensively copied into read-only views.

## MVP Boundaries

- Local execution only
- Synchronous and sequential worker trials
- No networking, remote workers, or hosted services
- No concurrency, threads, async execution, or process pools
- No database or resume-after-interruption behavior
- Seeded RandomSearch as the only required algorithm
- Two or more independently minimized or maximized objectives
- One immutable record for every attempted worker execution
- The complete Pareto Front, with no automatically selected universal winner
- GUI code optional and independent from optimizer logic

## Trunk-Based Development Workflow

The team is moving toward trunk-based development. `main` is the shared
integration branch, and implementation work occurs on short-lived,
timestamped working branches.

- Start each work session from an up-to-date `main`.
- Name working branches `work/YYYY-MM-DD-HHmm-short-topic`, using the local
  date and time when the branch is created. For example:
  `work/2026-07-28-1430-config-loader`.
- Keep each branch focused and merge it back into `main` the same day when
  practical. Begin the next day's work from the updated `main`.
- Commit frequently at meaningful, buildable checkpoints. Use clear commit
  messages that identify what changed or what remains incomplete.
- Do not force-push or rewrite shared working-branch history. Merge without
  squashing so checkpoint commits remain recoverable if a later change breaks.
- Before merging, run the relevant tests and the global monolith checker, then
  update the scratch memory in `AGENTS.md`.
- Keep `main` stable. Do not begin feature work directly on `main`, and do not
  allow working branches to become long-lived alternate integration branches.
- Retain merged timestamped branches as named checkpoints until the team
  agrees on a cleanup policy.

## Development Verification

Run the current test suite with the required interpreter:

```powershell
py -3.13 -m unittest discover -s tests -p "test_*.py"
```

Run the repository-wide source hygiene check:

```powershell
py -3.13 tools\check_monoliths.py
```

The application does not yet have a runnable optimizer entry point. The next
approved implementation step will add either configuration loading or the
one-trial vertical slice; this README should be updated as planned modules
become real.
