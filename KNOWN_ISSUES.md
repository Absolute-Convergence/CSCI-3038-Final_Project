# Known Issues

Confirmed, reproduced bugs in merged code -- not a wishlist.

## Open

Found during a dedicated adversarial bug-hunt pass (six parallel reviewers,
one per module cluster), immediately after a separate coverage-closing pass
brought the suite to 99% line/branch coverage with no bugs found. Every
issue below was independently reproduced by running the actual code, not
inferred from reading alone.

### Invalid configuration paths can leak a raw `ValueError` on Windows

`config_loader.py` translates failures from `Path.resolve()` into
`ConfigurationError`, but `_read_json()` catches only `OSError` and
`UnicodeError` around `Path.open()`. On Windows with Python 3.13.14,
`Path("bad\\0path.json").resolve()` succeeds and `Path.open()` then raises
`ValueError: embedded null character`. The raw exception escapes
`load_configuration()` instead of becoming the documented configuration
diagnostic. Reproduced by
`ConfigurationLoaderTests.test_rejects_unresolvable_path`: the same test
passes when the host rejects the path during `resolve()` but errors on Windows
when the rejection happens during `open()`. This also makes the proposed
Windows/Ubuntu/macOS package test matrix fail on Windows.

### A `KeyboardInterrupt` during report-writing mislabels a completed run as cancelled

`controller.py`'s `_UNSAFE_TO_CANCEL_STATES` covers `EXECUTING` and
`RECORDING` but not `FINALIZING`. If a `KeyboardInterrupt` lands inside
`self._reporter.write(self.result)`, it isn't a `ReportingError` so
`_finalize()`'s own except clause doesn't catch it; it propagates to the
outer handler, which unconditionally sets
`self.termination_reason = "user_cancelled"` and calls `_finalize()` again.
Since the already-correct result's `termination_reason` now differs from
the new one, the rebuild guard discards the correct result and replaces it
with one labeled `status="cancelled"` -- the reporter gets invoked twice,
once with each. Reproduced: a run whose one trial genuinely completed
(`history` length 1) ends with `result.status == "cancelled"` and
`reporter.calls == 2` after a `KeyboardInterrupt` on the first write.

### A failed per-trial `mkdir()` bypasses the checkpoint-failure safety net

`persistence.py`'s `RunDirectory.trial_directory()` calls
`directory.mkdir(parents=True, exist_ok=True)` with no try/except, unlike
every other filesystem operation in the class. `write_diagnostics()` calls
`trial_directory()` *outside* its own `except OSError -> CheckpointError`
block. `controller.py`'s `RECORDING` state only catches `CheckpointError`
around this call, specifically so `_finalize()` can still run and write
partial evidence on failure. Reproduced: forcing `trial_directory()` to
raise `OSError` on its second call makes the raw `OSError` escape
`controller.run()` entirely -- the run directory ends up with **no**
`summary.txt`, **no** `history.csv`, none of the partial-evidence
artifacts the design explicitly promises for a fatal checkpoint failure.

### An unanticipated exception during EXECUTING/RECORDING strands the controller and masks itself on retry

`candidate`/`metrics_path`/`execution_result` are local variables inside
`controller.py`'s `run()`, reset to `None` at the top of every call, while
`self.state` is an instance attribute that survives across calls. If
`runner.execute()` or `build_trial_record()` raises something other than
`KeyboardInterrupt` (e.g. a collaborator bug producing a malformed
execution-result dict), the exception correctly propagates out of `run()`
-- but `self.state` is left at `EXECUTING`/`RECORDING`. Reproduced: calling
`run()` again after catching that first exception raises an unrelated
`RuntimeError: EXECUTING reached without a candidate` instead of the real
error, and the in-flight trial attempt is silently dropped -- never
recorded, never checkpointed, never counted -- violating the "every
authorized attempt must be recorded" design invariant.

### Huge integer bounds raise `OverflowError` instead of `ValueError`

`models.py`'s `ParameterDefinition._validate_float_domain` (float bounds)
and `WorkerSpec.__post_init__` (`timeout_seconds`) both call
`math.isfinite(float(value))` after only checking `_is_number(value)`,
which accepts a plain `int`. Python's `float()` on an arbitrarily large
`int` raises `OverflowError`, not `ValueError`. Reproduced:
`ParameterDefinition("x", ParameterKind.FLOAT, minimum=0.0,
maximum=10**400)` raises `OverflowError: int too large to convert to
float` instead of the intended "must be finite" `ValueError`.
`config_loader.py` happens to also catch `OverflowError`, so config-loading
survives but surfaces the raw Python message instead of the intended one;
any direct/programmatic construction outside `config_loader` gets an
uncaught `OverflowError`.

### Categorical validation accepts a float that merely equals an int choice

`candidate_validation.py`'s `_validate_value()` checks categorical values
with `value in definition.choices`, which uses `==` -- and `8.0 == 8` is
`True` in Python. Reproduced: a candidate with `batch_size=8.0` against
`choices=(8, 16, 32)` passes `validate_candidate()` and comes back still a
`float`, not the declared `int`.

### `WorkerSpec.metrics_argument` accepts a bare `"--"` with no flag name

`models.py` only checks `metrics_argument.startswith("--")`, never that
anything follows it. Reproduced:
`WorkerSpec(command=(...), metrics_argument="--", timeout_seconds=1.0)`
constructs successfully. In most CLI/argparse conventions a bare `"--"` is
the "end of options" sentinel, not a flag -- this "valid" `WorkerSpec`
would silently produce a broken worker invocation rather than being
rejected at config time.

### Windows-style absolute paths get mangled when a config is loaded on POSIX (and vice versa)

`config_loader.py`'s `_resolve_command_part()` uses `Path.is_absolute()`,
which is platform-dependent. A Windows path like `"C:\\Python313\\python.exe"`
loaded on macOS/Linux is not recognized as absolute, so it gets joined
under the configuration directory and `.resolve()`d into a bogus,
nonexistent path instead of being left alone or rejected. Reproduced:
loading a config with that worker command on this (POSIX) machine resolves
it to `/private/.../tmpXXXX/C:\Python313\python.exe`. Every trial then
fails with a confusing `launch_failed` and no indication the real cause is
a path-mangling bug. The repo already ships a Windows-flavored example
config (`examples/iris_torch/iris_config.json`), so mixed-platform config
use is in scope.

### `Reporter.write()` isn't atomic as a group

`reporting.py`'s `write()` performs four separate atomic writes in
sequence (`resolved_config.json`, `pareto_front.csv`, `summary.txt`,
`pareto_front.png`) with no rollback across the group. Each individual
`os.replace()` is atomic, but a failure partway through leaves whatever
already succeeded on disk. Reproduced: forcing the third write
(`summary.txt`) to fail on a fresh run directory leaves
`resolved_config.json` and `pareto_front.csv` present while `summary.txt`
and `pareto_front.png` are simply absent -- a partial, inconsistent
artifact bundle, not the all-or-nothing guarantee the "authoritative
result artifacts" framing implies.

### `checkpoint()`/`write_diagnostics()` only catch `OSError`, so other failures leak a temp file and bypass `CheckpointError`

`persistence.py`'s `checkpoint()` builds CSV rows with
`record.parameters[parameter.name]` (direct indexing, not `.get()`) inside
the same `try` block whose `except` clause only matches `OSError`.
Reproduced: a record missing a declared parameter raises a bare `KeyError`
that isn't caught, leaving an orphaned `.tmp` file in the run directory and
propagating an unhandled exception instead of the documented
`CheckpointError` -> fatal-finalize path. Currently unreachable through the
sanctioned pipeline (candidates are validated before a `TrialRecord` is
built), so this is a robustness/defense-in-depth gap rather than a live
production crash today -- but it's a real mismatch between the atomic-write
contract's stated intent and what's actually handled. The same narrow
`except OSError` pattern exists in `_atomic_write_text()`.

## Resolved

### `AlgorithmSpec` accepted a negative `seed`; the CLI crashed instead of exiting cleanly

`models.py`'s `AlgorithmSpec.__post_init__` only checked that `seed` was an
`int`, never its sign. `config_loader.py` therefore accepted
`"algorithm": {"seed": -1, ...}` as a valid configuration.
`RandomSearch.__init__`/`NSGA2.__init__` did reject a negative seed, but
with a bare `ValueError("seed cannot be negative")`, raised from inside
`create_algorithm()` in `application.py`'s `initialize_application()`. That
call sat outside any `try` block `cli.py`'s `main()` could catch (its
except clauses are `KeyboardInterrupt`, `ConfigurationError`, `(OSError,
ReportingError)` -- a bare `ValueError` matched none of them). Reproduced:
running `main()` against a real config with `seed=-1` raised
`ValueError: seed cannot be negative` straight out of `main()` -- no exit
code was ever returned, the process crashed with a raw traceback. Covered
by `tests/test_models.py::test_algorithm_spec_rejects_negative_seed` and
`tests/test_config_loader.py::test_rejects_a_negative_seed_as_a_clean_configuration_error`.

PR #24 added the same "seed cannot be negative" check to `AlgorithmSpec`
itself, the earliest point in the config-driven path, so
`config_loader.py`'s existing `ValueError` -> `ConfigurationError`
wrapping catches it for free -- the same clean exit code 2 every other
malformed config value already gets. Merged into `main` as `faae4d8`.

### NSGA2 crowding distance was order-dependent when an objective was tied

`search/nsga2.py`'s `_crowding_distances()` marked `sorted_tier[0]`/
`sorted_tier[-1]` as boundary (`float("inf")`) unconditionally, before
checking whether that objective's range was zero (fully tied). When an
objective was tied, Python's stable sort just preserves input order, so
whichever records happened to land first/last in the *input* -- not the
actual extremes -- got wrongly marked as boundary points. Reproduced: the
same 4 records with one tied objective produced `{0: inf, 1: inf, 2: inf,
3: inf}` in one input order and the mathematically-correct `{0: inf,
1: 0.625, 2: 0.625, 3: inf}` in another, differing only by argument order.
This biased NSGA2's parent selection toward incidental list position
instead of genuine objective diversity whenever any tier had a tied
objective (common with quantized or early-generation-plateau metrics).
Covered by `tests/test_nsga2.py`,
`test_tied_objective_never_marks_a_boundary_regardless_of_input_order` --
confirmed to fail against the pre-fix code and pass against the fix.

PR #23 moved the `objective_range == 0` check before the boundary
assignment, so a tied objective is now skipped entirely and contributes
nothing, regardless of input order. Merged into `main` as `341abdb`. A
follow-up aggressive re-read of the rest of `nsga2.py` found no further
bugs.

### CLI run artifacts appeared as untracked source files

The CLI defaults to the repository-root `runs/` directory, but that generated
output directory was initially absent from `.gitignore`. A normal 20-trial Iris
run therefore appeared as dozens of untracked CSV, PNG, JSON, summary, and
diagnostic files. The root-only `/runs/` rule now excludes current and future
local run artifacts without deleting them or ignoring unrelated nested
directories named `runs`.

### `runner.py`: `process_failed` trials lost their error message

At the start of this work, merged `main` used `subprocess.run()` and discarded
the crashed worker's captured output, so `error_message` was `None` on a
`process_failed` trial even when the worker printed a real traceback.
Reproduced by running the Iris worker with `batch_size=0`: exit code 1 and no
diagnostic summary. Covered by
`tests/integration/test_full_pipeline_real_worker.py`, in
`test_real_worker_crash_flows_through_as_process_failed`.

PR #14 updated that integration assertion to the approved bounded diagnostic
behavior.

Required remediation:

- Put a bounded, human-readable immediate cause in `error_message` for a
  nonzero exit.
- Preserve full captured stdout/stderr in trial-local diagnostic files when
  the run-directory boundary is available.
- Do not copy unbounded worker stderr into `history.csv`; the TDS requires
  bounded history diagnostics and trial-local storage for full output.

The implemented policy uses a 1,000-character maximum, exit context plus the
last nonblank stderr line, UTF-8 replacement decoding,
`stdout.txt`/`stderr.txt`, and no speculative redaction. PR #14 returns complete
private streams from Runner and persists them through `RunDirectory`; no
unbounded stream content enters `history.csv`. Its reviewed tree reached
`main` through the final stacked-merge integration repair.
