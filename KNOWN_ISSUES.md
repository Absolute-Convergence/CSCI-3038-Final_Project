# Known Issues

Confirmed, reproduced defects and their remediation status. This is not a
wishlist.

## Open

No confirmed defects remain open at the package-release remediation checkpoint.

## Resolved

### The GUI results-folder button called a Windows-only API

The optional GUI used `os.startfile()` unconditionally after an optimization
completed, raising `AttributeError` on macOS and Linux. The folder launcher now
dispatches to `os.startfile` on Windows, `open` on macOS, and `xdg-open` on
Linux. macOS and Linux launchers use nonblocking argument-list subprocesses;
failure leaves the results intact and reports their resolved location. Focused
tests cover all three platform branches and launcher failure. The GUI remains
outside the separately verified core-package compatibility boundary.

### Invalid configuration paths leaked a raw `ValueError` on Windows

On Windows, a null byte could survive `Path.resolve()` and fail later in
`Path.open()`. `_read_json()` now translates that `ValueError` into the
same location-aware `ConfigurationError` used for other invalid paths. The
previously failing Windows regression test now passes.

### A finalization interrupt mislabeled a completed run as cancelled

`FINALIZING` is now an unsafe cancellation state. A `KeyboardInterrupt`
during report writing propagates without replacing the authoritative completed
result, changing its termination reason, or silently invoking the reporter
twice. A later explicit retry reuses the same immutable result.

### A failed per-trial directory creation bypassed checkpoint handling

`RunDirectory.trial_directory()` now converts an `OSError` into
`CheckpointError`. The controller retains any in-memory record and reaches
fatal finalization instead of leaking a raw filesystem error and losing the
partial result.

### Unexpected EXECUTING or RECORDING failures stranded the controller

Mel approved `execution_status="internal_error"` for this boundary.
Unexpected collaborator exceptions now produce one immutable, bounded-
diagnostic `TrialRecord` for the authorized attempt, checkpoint it when
possible, and finalize the run with `fatal_error`. Retrying the stopped
controller returns the same result instead of masking the original failure.

### Huge integer values leaked `OverflowError` from numeric validation

Finite-number validation now treats integers too large for `float()` as
nonfinite input. Float parameter bounds, worker timeouts, and float candidate
values reject them through their documented `ValueError` or
`CandidateValidationError` boundaries.

### Categorical validation confused equal integer and float values

Categorical membership now requires both value equality and exact type
identity. For example, `8.0` no longer satisfies an integer choice of `8`.

### A bare `"--"` was accepted as the worker metrics argument

`WorkerSpec` now requires at least one character after the double-hyphen
prefix, so the end-of-options sentinel is rejected during configuration
validation.

### Foreign-platform absolute worker paths were mangled

Configuration loading now recognizes Windows and POSIX absolute-path syntax
independently of the host. An absolute path written for another operating
system is rejected with a location-aware `ConfigurationError` instead of
being rewritten into a bogus native path.

### Report writing was assumed to be atomic as a group

Mel clarified that atomicity is guaranteed per file, not across the complete
report collection. Each file is still atomically replaced. A reporting failure
exits nonzero and means the collection is incomplete; already committed files
remain individually valid. The root-level artifact layout is unchanged.

### Non-filesystem checkpoint failures leaked temporary files

`checkpoint()` now cleans its temporary file and translates any ordinary
serialization failure, including a missing declared parameter `KeyError`,
into `CheckpointError`. Diagnostic persistence similarly translates ordinary
write failures, while interruption still propagates after temporary cleanup.


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
