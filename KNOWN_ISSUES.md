# Known Issues

Confirmed, reproduced bugs in merged code -- not a wishlist.

## Open

_(no confirmed open defects at this checkpoint)_

## Resolved

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
