# Known Issues

Confirmed, reproduced bugs in merged code -- not a wishlist.

## Open

### `runner.py`: `process_failed` trials lose their error message

`execute()` captures the crashed worker's stdout/stderr
(`capture_output=True`) but never reads it before returning, so
`error_message` is always `None` on a `process_failed` trial even when
the worker printed a real traceback. Reproduced by running the Iris
worker with `batch_size=0` (a real `DataLoader` crash): exit code 1,
`error_message` still `None`. Covered by
`tests/integration/test_full_pipeline_real_worker.py`, in
`test_real_worker_crash_flows_through_as_process_failed`.

The integration test currently asserts `None` to preserve and expose the
reproduced behavior; it must be updated when the defect is fixed.

Required remediation:

- Put a bounded, human-readable immediate cause in `error_message` for a
  nonzero exit.
- Preserve full captured stdout/stderr in trial-local diagnostic files when
  the run-directory boundary is available.
- Do not copy unbounded worker stderr into `history.csv`; the TDS requires
  bounded history diagnostics and trial-local storage for full output.

The exact summary, truncation, and redaction rules affect a controlled
diagnostic contract and require human agreement before implementation. The
existence of a useful bounded cause is already required by TDS section 10.4.

## Resolved

_(move fixed entries here with the PR number instead of deleting them)_
