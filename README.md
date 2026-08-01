# Black Box Optimizer

CSCI 3038 final project by The Snek People.

Black Box Optimizer is a local Python 3.13.14 application under development for
tuning an external worker program without importing or inspecting that
worker's internal code. It sends candidate parameters through command-line
flags, receives one row of numerical metrics through a trial-specific CSV
file, and will return the complete non-dominated Pareto Front across multiple
objectives when the remaining MVP composition work is complete.

`main` currently implements immutable configuration and candidate models, the
validated JSON loader, seeded RandomSearch, synchronous worker execution,
metrics parsing, immutable trial records, append-only in-memory history,
maximum-trial stop decisions, and an external PyTorch Iris worker. A real
integration test exercises those components together through actual worker
subprocesses. The application controller, durable run persistence, full Pareto
evaluation, final results, reporting, and command-line entry point are not on
`main` yet, so the repository does not yet provide a runnable optimizer.

See [docs/architecture-baseline.md](docs/architecture-baseline.md) for the
controlling foundation contracts and MVP boundaries.

The current planning guidance is the aligned Draft v0.2 document set in
[`docs/planning_baseline_v02`](docs/planning_baseline_v02). It remains pending
team ratification and technical approval. Draft v0.1 is preserved unchanged in
[`docs/planning_baseline_v01`](docs/planning_baseline_v01) as the historical
source baseline.

Repository decisions and operating guidance in [AGENTS.md](AGENTS.md), together
with the approved architecture baseline, control when the draft planning set
conflicts with merged code or a recorded human decision. Public-contract
differences must still be reconciled through change control; implementation
drift recorded in scratch notes does not silently amend a controlled contract.

## Target MVP Runtime Flow

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

## Project Structure and Status

Items marked `implemented` exist now. Other modules show the intended
responsibility boundaries and will be added in later, approved changes.

```text
CSCI-3038-Final_Project/
|-- AGENTS.md                         # repository rules and scratch memory
|-- KNOWN_ISSUES.md                   # reproduced defects in merged code
|-- README.md                         # project overview and development map
|-- requirements.txt                 # current NumPy and example-worker deps
|-- source_hygiene.json               # global source-file hygiene settings
|-- black_box_optimizer/
|   |-- __init__.py                   # implemented public model exports
|   |-- __main__.py                   # planned module entry point
|   |-- cli.py                        # planned CLI composition
|   |-- models.py                     # implemented immutable foundation types
|   |-- config_loader.py              # implemented JSON parsing and validation
|   |-- controller.py                 # planned sequential lifecycle governor
|   |-- runner.py                     # implemented synchronous subprocess boundary
|   |-- metrics.py                    # implemented one-row CSV parser
|   |-- records.py                    # implemented TrialRecord construction
|   |-- history.py                    # implemented append-only TrialHistory
|   |-- persistence.py                # planned durable history snapshots
|   |-- stop_policy.py                # implemented maximum-trial decisions
|   |-- pareto.py                     # planned eligibility and dominance
|   |-- results.py                    # planned ParetoFront/OptimizationResult
|   |-- reporting.py                  # planned result export boundary
|   `-- search/
|       |-- base.py                   # implemented search protocol/results
|       |-- registry.py               # implemented built-in algorithm registry
|       `-- random_search.py          # implemented seeded RandomSearch
|-- examples/
|   |-- __init__.py                   # implemented, makes examples importable
|   `-- iris_torch/
|       |-- __init__.py               # implemented package marker
|       |-- iris_config.json          # implemented example configuration
|       |-- iris-data.csv             # implemented bundled Iris dataset
|       `-- worker.py                 # implemented external PyTorch worker
|-- tests/
|   |-- test_models.py                # implemented foundation-model tests
|   |-- test_metrics.py               # implemented metrics-parser tests
|   |-- test_records.py               # implemented trial-record tests
|   |-- test_stop_policy.py           # implemented stop-policy tests
|   |-- test_history.py               # implemented trial-history tests
|   |-- test_runner.py                # implemented execute tests
|   |-- test_search_base.py           # implemented ProposalResult tests
|   |-- test_search_registry.py       # implemented algorithm-registry tests
|   |-- test_random_search.py         # implemented RandomSearch tests
|   |-- test_worker.py                # implemented Iris worker tests
|   |-- test_check_monoliths.py       # implemented hygiene-checker tests
|   |-- unit/                         # planned focused unit tests
|   |-- integration/
|   |   |-- test_one_trial_slice.py   # implemented module-level slice
|   |   `-- test_full_pipeline_real_worker.py
|   |                                   # implemented real subprocess slice
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

## Implemented Public Surfaces

`load_configuration(path)` reads one UTF-8 JSON project file, validates its
complete structure before any trial can run, resolves relative worker script
or executable paths from the configuration file's directory, and returns an
immutable `ProjectConfiguration`. Invalid input raises `ConfigurationError`
with one or more location-aware issues.

The `black_box_optimizer` package root currently exports:

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
- `ConfigurationError`
- `load_configuration`

These are frozen dataclasses or string enums. Ordered collections use tuples,
and candidate mappings are defensively copied into read-only views.

Additional implemented surfaces are imported from their owning modules:

- `black_box_optimizer.metrics.read_trial_metrics`
- `black_box_optimizer.records.TrialRecord` and `build_trial_record`
- `black_box_optimizer.history.TrialHistory`
- `black_box_optimizer.stop_policy.StopDecision` and `StopPolicyEvaluator`
- `black_box_optimizer.runner.execute`
- `black_box_optimizer.search.base.ProposalResult` and `SearchAlgorithm`
- `black_box_optimizer.search.registry.create_algorithm`
- `black_box_optimizer.search.random_search.RandomSearch`

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for reproduced defects in merged code.

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
- After a verified merge, create and push an annotated checkpoint tag named
  `checkpoint/main-<topic>-YYYY-MM-DD-HHmm` on the merge commit.
- Delete the completed working branch locally and remotely. The non-squash
  merge and checkpoint tag preserve its history; merged branches do not remain
  active.

## Development Verification

Install the current dependencies into the same Python 3.13 interpreter used to
run the tests. NumPy supports seeded RandomSearch, while PyTorch belongs only to
the external Iris example and its tests:

```powershell
py -3.13 -m pip install -r requirements.txt
```

Run the current test suite with the required interpreter:

```powershell
py -3.13 -m unittest discover -s tests -p "test_*.py"
```

Run the repository-wide source hygiene check:

```powershell
py -3.13 tools\check_monoliths.py
```

At the 2026-08-01 documentation checkpoint, all 134 tests pass in the local
Python 3.11 environment with dependencies installed, and the source-hygiene
check passes. Python 3.13.14 is installed locally but still needs the project
dependencies before the required-interpreter suite can be confirmed there.

The application does not yet have a runnable optimizer entry point. On `main`,
the next composition boundary is the sequential application controller plus
durable run persistence. Full Pareto evaluation, result construction,
reporting, and CLI composition follow that boundary. An in-progress remote
work branch is not part of the `main` implementation status described here.
