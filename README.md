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
|   |-- zdt1_benchmark/
|   |   |-- synthetic_config.json     # dependency-free smoke-test config
|   |   `-- compare_search_algorithms.py
|   `-- paper_airplane/
|       |-- __init__.py               # package marker
|       |-- paper_airplane_config.json       # example configuration (Windows)
|       |-- paper_airplane_config_unix.json  # same example, macOS/Linux worker command
|       `-- paper_airplane_worker.py  # fun demo worker, known-optimal Pareto front
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

`worker.command` in a project configuration is the exact command that
already runs your worker on your own machine, so it is inherently
OS-specific -- Hyperloop never inspects or launches Python itself, it
just execs what you give it. Every example below is shown for both
Windows and macOS/Linux; use whichever matches your machine.

Install the released distribution and invoke its dedicated command with a
project configuration:

```powershell
py -3.13 -m pip install hyperloop-optimizer
hyperloop-optimizer path\to\config.json --output-dir runs
```

```bash
python3 -m pip install hyperloop-optimizer
hyperloop-optimizer path/to/config.json --output-dir runs
```

The existing module command remains supported:

```powershell
py -3.13 -m black_box_optimizer path\to\config.json --output-dir runs
```

```bash
python3 -m black_box_optimizer path/to/config.json --output-dir runs
```

From a source checkout, install the core package in editable mode. Install the
Iris dependency separately only when running that example. `iris_config.json`
uses Windows' `py` launcher for its worker command; there is currently no
macOS/Linux example configuration for Iris specifically, so on those
platforms either edit `worker.command` to `["python3", "iris_worker.py"]`
yourself or use the Paper Airplane example below instead, which ships a
config for both platforms already:

```powershell
py -3.13 -m pip install -e .
py -3.13 -m pip install -r requirements-iris.txt
py -3.13 -m black_box_optimizer `
  examples\iris_torch\iris_config.json `
  --output-dir runs
```

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements-iris.txt
python3 -m black_box_optimizer \
  examples/iris_torch/iris_config.json \
  --output-dir runs
```

The source checkout also includes the Paper Airplane example: a fun,
near-instant worker (pure arithmetic, no ML training) with a genuine,
mathematically known-optimal Pareto front, useful for demos or for
seeing real multi-objective search results in seconds instead of
minutes. It ships a working config for both platforms already:

```powershell
py -3.13 -m black_box_optimizer `
  examples\paper_airplane\paper_airplane_config.json `
  --output-dir runs
```

```bash
python3 -m black_box_optimizer \
  examples/paper_airplane/paper_airplane_config_unix.json \
  --output-dir runs
```

The distribution also includes a near-instant synthetic worker for smoke
tests and search evaluation. It has its own installed command so worker
configuration never needs to guess which Python interpreter owns Hyperloop:

```powershell
hyperloop-synthetic-worker --help
```

```bash
hyperloop-synthetic-worker --help
```

The source checkout includes a ready-to-run configuration:

```powershell
hyperloop-optimizer `
  examples\zdt1_benchmark\synthetic_config.json `
  --output-dir runs
```

```bash
hyperloop-optimizer \
  examples/zdt1_benchmark/synthetic_config.json \
  --output-dir runs
```

That four-trial configuration is an installation and artifact smoke test, not
a search-quality claim. Search efficacy is measured separately against ZDT1's
known optimal front with repeated seeds and dominated hypervolume. The default
comparison launches 10,000 real worker subprocesses: two algorithms, ten seeds
per algorithm, and 500 trials per seeded run.

```powershell
py -3.13 -m examples.zdt1_benchmark.compare_search_algorithms
```

```bash
python3 -m examples.zdt1_benchmark.compare_search_algorithms
```

Use one or five seeds for the shorter 1,000- or 5,000-trial tiers while
keeping 500 trials in each independent run. The unchanged default is the
10,000-trial release-evidence tier:

```powershell
py -3.13 -m examples.zdt1_benchmark.compare_search_algorithms --seeds 1
py -3.13 -m examples.zdt1_benchmark.compare_search_algorithms --seeds 5
py -3.13 -m examples.zdt1_benchmark.compare_search_algorithms
```

```bash
python3 -m examples.zdt1_benchmark.compare_search_algorithms --seeds 1
python3 -m examples.zdt1_benchmark.compare_search_algorithms --seeds 5
python3 -m examples.zdt1_benchmark.compare_search_algorithms
```

Independent seed/algorithm runs can execute concurrently instead of one
at a time -- this only affects wall-clock time, never results, since
each run is fully independent:

```bash
python3 -m examples.zdt1_benchmark.compare_search_algorithms --jobs 8
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

The Hyperloop core package is verified on Python 3.13.14 with GitHub-hosted
Windows, Ubuntu, and macOS runners. Each platform installs the same validated
wheel before running the core-compatible test suite, dependency validation,
source hygiene check, and installed synthetic-worker smoke test. The optional
Tkinter GUI and PyTorch Iris example are outside that package compatibility
claim.

## Bring Your Own Worker

The three examples above (Iris, ZDT1, Paper Airplane) each pair a config
file with a worker script. To optimize your own program instead, you
need both pieces yourself. This section documents the exact contract
for each, verified directly against the loader and runner code -- not
just inferred from the examples.

### The worker contract

Your worker can be written in any language, as long as it's an
executable command. Hyperloop never imports or inspects it -- it just
runs your `worker.command` as a subprocess, once per trial, with two
kinds of arguments appended automatically:

1. **One `--<parameter-name> <value>` flag per declared parameter**,
   with underscores converted to dashes. A parameter named
   `learning_rate` arrives as `--learning-rate 0.01`.
2. **Your configured metrics flag**, pointing at a file path your
   worker must write its results to. If `metrics_argument` is
   `"--metrics-out"`, your worker receives `--metrics-out /some/path.csv`.

Your worker must write that file as a CSV with **exactly two rows**: a
header row naming each metric, then one data row of numeric values.
Nothing else -- no extra rows, no missing header/data row.

```csv
validation_accuracy,validation_loss
0.94,0.18
```

A worker that exits with a nonzero code, times out, or writes anything
else is recorded as a failed trial rather than crashing the run -- see
`examples/iris_torch/iris_worker.py` or
`examples/paper_airplane/paper_airplane_worker.py` for complete,
minimal reference implementations.

### The config schema

A project configuration is one JSON file with four required top-level
sections: `worker`, `optimization`, `algorithm`, `stop_policy`.

```json
{
  "worker": {
    "command": ["python3", "your_worker.py"],
    "metrics_argument": "--metrics-out",
    "timeout_seconds": 60.0
  },
  "optimization": {
    "parameters": [
      {
        "name": "learning_rate",
        "kind": "float",
        "minimum": 0.0001,
        "maximum": 0.1
      },
      {
        "name": "batch_size",
        "kind": "categorical",
        "choices": [8, 16, 32]
      }
    ],
    "objectives": [
      { "metric_name": "validation_accuracy", "direction": "maximize" },
      { "metric_name": "validation_loss", "direction": "minimize" }
    ]
  },
  "algorithm": {
    "name": "nsga2"
  },
  "stop_policy": {
    "max_trials": 150
  }
}
```

- **`worker.command`** -- an array of strings, the exact command that
  already runs your worker on your machine (see the OS-specific note
  above). A relative path in it is resolved relative to the config
  file's own directory, not your current working directory.
- **`worker.metrics_argument`** -- the flag name your worker expects
  for its output file path.
- **`worker.timeout_seconds`** -- how long one trial is allowed to run
  before being killed and recorded as `timed_out`.
- **`optimization.parameters`** -- each needs `name` and `kind`. `kind`
  is `"float"`, `"integer"`, or `"categorical"`. Float/integer
  parameters also need `minimum`/`maximum`; categorical parameters need
  a `choices` array instead.
- **`optimization.objectives`** -- each needs `metric_name` (must match
  a column your worker's metrics CSV writes) and `direction`, either
  `"maximize"` or `"minimize"`. At least two objectives are required --
  this is a multi-objective optimizer.
- **`algorithm.name`** -- `"random_search"` or `"nsga2"`. NSGA2 needs a
  realistic trial budget to show its advantage: on this project's own
  ZDT1 benchmark it wasn't statistically distinguishable from
  random_search below roughly 150 trials, and its final quality kept
  improving well past that (see
  [docs/decisions/2026-08-04-nsga2-evolution-and-evaluation.md](docs/decisions/2026-08-04-nsga2-evolution-and-evaluation.md)
  for the full data). For genuinely small budgets, random_search is the
  safer default.
- **`algorithm.seed`** -- optional. If omitted, a fresh seed is
  generated automatically and reported back at the end of the run
  (printed to the terminal and saved in `resolved_config.json`) --
  supply that same value later to reproduce the exact run.
- **`stop_policy.max_trials`** -- a positive integer, the hard ceiling
  on how many trials this run will attempt.

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

At the package-remediation checkpoint, 394 tests pass under the
course-required Python 3.13.14 interpreter. The suite includes real
controller-to-Iris runs, a synthetic-worker subprocess test, an end-to-end
module CLI test, repeated-run isolation coverage, and packaging-boundary
checks. The source-hygiene check passes across 62 source files; line-length
advisories remain non-failing design guidance.
