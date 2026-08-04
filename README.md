# Hyperloop

CSCI 3038 final project by The Snek People.

Hyperloop is a local Python 3.13.14 black-box optimizer for
tuning an external worker program without importing or inspecting that
worker's internal code. It sends candidate parameters through command-line
flags, receives one row of numerical metrics through a trial-specific CSV
file, and returns the complete non-dominated Pareto Front across multiple
objectives.

The implemented MVP includes immutable configuration and candidate models,
validated JSON loading, declared-domain proposal validation, seeded Random
Search and NSGA-II, synchronous cancellable worker execution, immutable trial
records, append-only history, lifecycle control, atomic persistence, complete
mixed-direction Pareto evaluation, immutable results, reporting, and module
and installed command-line entry points. The external PyTorch Iris worker
remains outside the distribution. A dependency-free ZDT1 synthetic worker is
distributed in a separate namespace so Hyperloop can still treat it as an
opaque subprocess.

See [docs/architecture-baseline.md](docs/architecture-baseline.md) for the
controlling foundation contracts and MVP boundaries.

See [docs/gui-handoff-next-steps.md](docs/gui-handoff-next-steps.md) for the
recommended handoff from Charles's Tkinter GUI to Hyperloop's existing CLI and
run artifacts.

See [docs/package-release-checklist.md](docs/package-release-checklist.md) for
the single-package v0.1.1 release boundary, audit evidence, and publication
gates.

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

## MVP Runtime Flow

```text
JSON project configuration
          |
          v
Configuration loader -> immutable ProjectConfiguration
          |
          v
Random Search or NSGA-II -> immutable CandidateConfiguration
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

Items marked `implemented` exist in the completed implementation stack. The GUI
remains an optional, independent presentation layer.

```text
CSCI-3038-Final_Project/
|-- AGENTS.md                         # repository rules and scratch memory
|-- KNOWN_ISSUES.md                   # reproduced defects in merged code
|-- README.md                         # project overview and development map
|-- pyproject.toml                    # package metadata and installed CLI
|-- requirements.txt                 # core optimizer/reporting dependencies
|-- requirements-iris.txt            # external Iris worker dependency
|-- requirements-gui.txt             # optional source GUI dependency
|-- requirements-release.txt         # build and publication tools
|-- source_hygiene.json               # global source-file hygiene settings
|-- black_box_optimizer/
|   |-- __init__.py                   # implemented public model exports
|   |-- __main__.py                   # implemented module entry point
|   |-- cli.py                        # implemented CLI composition
|   |-- application.py                # implemented initialization/composition
|   |-- models.py                     # implemented immutable foundation types
|   |-- config_loader.py              # implemented JSON parsing and validation
|   |-- controller.py                 # implemented lifecycle governor
|   |-- runner.py                     # implemented Popen subprocess boundary
|   |-- metrics.py                    # implemented one-row CSV parser
|   |-- records.py                    # implemented TrialRecord construction
|   |-- history.py                    # implemented append-only TrialHistory
|   |-- persistence.py                # implemented history + trial artifacts
|   |-- stop_policy.py                # implemented maximum-trial decisions
|   |-- pareto.py                     # implemented eligibility/dominance/sweep
|   |-- results.py                    # implemented immutable result contracts
|   |-- reporting.py                  # implemented result export boundary
|   `-- search/
|       |-- base.py                   # implemented search protocol/results
|       |-- registry.py               # implemented built-in algorithm registry
|       |-- random_search.py          # implemented seeded Random Search
|       `-- nsga2.py                  # implemented seeded NSGA-II
|-- hyperloop_workers/
|   |-- __init__.py                   # separate bundled-worker namespace
|   `-- synthetic_worker.py           # dependency-free ZDT1 worker
|-- examples/
|   |-- __init__.py                   # implemented, makes examples importable
|   |-- iris_torch/
|   |   |-- __init__.py               # implemented package marker
|   |   |-- iris_config.json          # implemented example configuration
|   |   |-- iris-data.csv             # implemented bundled Iris dataset
|   |   `-- iris_worker.py            # external PyTorch worker
|   `-- zdt1_benchmark/
|       |-- synthetic_config.json     # dependency-free smoke-test config
|       `-- compare_search_algorithms.py
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
|   |-- test_nsga2.py                 # implemented NSGA-II tests
|   |-- test_iris_worker.py           # implemented Iris worker tests
|   |-- test_zdt1_benchmark.py        # synthetic worker/benchmark tests
|   |-- test_packaging.py             # distribution-boundary tests
|   |-- test_check_monoliths.py       # implemented hygiene-checker tests
|   |-- test_controller.py            # implemented ApplicationController tests
|   |-- test_pareto.py                # implemented full Pareto tests
|   |-- test_persistence.py           # implemented RunDirectory tests
|   |-- test_application.py           # implemented composition tests
|   |-- test_cli.py                   # implemented CLI status tests
|   |-- unit/                         # planned focused unit tests
|   |-- integration/
|   |   |-- test_one_trial_slice.py   # implemented module-level slice
|   |   |-- test_full_pipeline_real_worker.py
|   |   |-- test_application_real_worker.py
|   |   `-- test_cli_acceptance.py    # implemented real application slices
|   `-- fixtures/                     # implemented purpose-built workers
|-- docs/
|   |-- architecture-baseline.md      # implemented controlling baseline
|   |-- gui-handoff-next-steps.md     # recommended GUI integration sequence
|   |-- planning_baseline_v01/        # preserved historical planning baseline
|   `-- planning_baseline_v02/        # current aligned draft planning guidance
`-- tools/
    `-- check_monoliths.py            # implemented source hygiene checker
```

Workers remain outside `black_box_optimizer`. The installed
`hyperloop_workers.synthetic_worker` uses only the Python standard library.
PyTorch belongs only to the source-checkout Iris example and its tests; it is
not a dependency of the Hyperloop distribution.

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
| Persistence | Run/trial directories, diagnostics, and atomic history checkpoints | Ranking, evaluating, or interpreting metrics |
| Pareto evaluator | Eligibility, mixed-direction dominance, and complete front | Weighted scoring or worker execution |
| Reporter | Authoritative exports and explanatory visualization | Optimizer state or universal-winner selection |
| GUI | Optional presentation of authoritative results | Required optimizer logic or result mutation |

The intended dependency direction is inward toward immutable contracts. The
Runner will not import search or Pareto code; RandomSearch will not import the
Runner; and the optimizer package will never import a worker implementation.

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
- `ParetoFront`
- `OptimizationResult`
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
- `black_box_optimizer.controller.ApplicationController`
- `black_box_optimizer.pareto.is_eligible`
- `black_box_optimizer.persistence.RunDirectory` and `create_run_directory`
- `black_box_optimizer.reporting.Reporter`
- `black_box_optimizer.application.initialize_application`
- `black_box_optimizer.search.base.ProposalResult` and `SearchAlgorithm`
- `black_box_optimizer.search.registry.create_algorithm`
- `black_box_optimizer.search.random_search.RandomSearch`
- `black_box_optimizer.search.nsga2.NSGA2`

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for reproduced defects in merged code.

## Install and Run Hyperloop

Install the released distribution and invoke its dedicated command with a
project configuration:

```powershell
py -3.13 -m pip install hyperloop-optimizer
hyperloop-optimizer path\to\config.json --output-dir runs
```

The existing module command remains supported:

```powershell
py -3.13 -m black_box_optimizer path\to\config.json --output-dir runs
```

From a source checkout, install the core package in editable mode. Install the
Iris dependency separately only when running that example:

```powershell
py -3.13 -m pip install -e .
py -3.13 -m pip install -r requirements-iris.txt
py -3.13 -m black_box_optimizer `
  examples\iris_torch\iris_config.json `
  --output-dir runs
```

The distribution also includes a near-instant synthetic worker for smoke
tests and search evaluation. It has its own installed command so worker
configuration never needs to guess which Python interpreter owns Hyperloop:

```powershell
hyperloop-synthetic-worker --help
```

The source checkout includes a ready-to-run configuration:

```powershell
hyperloop-optimizer `
  examples\zdt1_benchmark\synthetic_config.json `
  --output-dir runs
```

Each invocation creates a unique `run_*` directory. It contains the resolved
configuration, atomically checkpointed `history.csv`, one directory per trial,
the complete Pareto CSV, a text summary, and a PNG showing the first two
declared objectives. Every recorded trial directory contains `stdout.txt` and
`stderr.txt`; `metrics.csv` exists only when the worker produced it. The
repository-root `runs/` and `optimizer_runs/` directories are ignored by Git
because these are local, generated run artifacts rather than source files.
Every invocation owns a new unique run directory, a fresh history, and a
Pareto front derived only from that run.

Atomicity is guaranteed per file, including every `history.csv` checkpoint
and each final report replacement. The collection of final report files is not
a single transaction: if reporting fails, the command exits nonzero and files
already committed remain valid individually, but the report collection is
incomplete and must not be treated as a completed bundle.

Normal completion and a no-eligible-trials result exit with code 0. Fatal
failure exits with code 1, invalid initialization/configuration exits with code
2, and user cancellation exits with code 130.

The core is locally verified on Windows. Automated core-package gates are
configured for Windows, Ubuntu, and macOS; support for all three should be
claimed only after those GitHub Actions jobs pass. The optional Tkinter GUI is
outside that package compatibility claim.

## MVP Boundaries

- Local execution only
- Synchronous and sequential worker trials
- No networking, remote workers, or hosted services
- No concurrency, threads, async execution, or process pools
- No database or resume-after-interruption behavior
- Seeded Random Search as the required baseline; seeded NSGA-II is also shipped
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

Install Hyperloop and the separate Iris test dependency into the same Python
3.13 interpreter used to run the complete repository suite:

```powershell
py -3.13 -m pip install -e .
py -3.13 -m pip install -r requirements-iris.txt
```

Run the current test suite with the required interpreter:

```powershell
py -3.13 -m unittest discover -s tests -p "test_*.py"
```

Run the repository-wide source hygiene check:

```powershell
py -3.13 tools\check_monoliths.py
```

At the package-preparation checkpoint, 291 tests pass under the
course-required Python 3.13.14 interpreter. The suite includes real
controller-to-Iris runs, a synthetic-worker subprocess test, an end-to-end
module CLI test, repeated-run isolation coverage, and packaging-boundary
checks. The source-hygiene check passes across 62 source files; line-length
advisories remain non-failing design guidance.
