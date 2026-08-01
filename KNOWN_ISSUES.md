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
`tests/integration/test_full_pipeline_real_worker.py::test_real_worker_crash_flows_through_as_process_failed`.

Fix: capture `process.stderr` into `error_message` on that branch, same
as `launch_failed`/`timed_out` already do.
(`runner.py`). Double check with group since `error_message` contents may be a
change controlled contract detail!

## Resolved

_(move fixed entries here with the PR number instead of deleting them)_
