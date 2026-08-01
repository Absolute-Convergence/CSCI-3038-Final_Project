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

Same root cause has a second consequence: TDS section 10.4 wants full
stdout/stderr preserved in trial-local files (`stdout.txt`/`stderr.txt`
next to each trial's `metrics.csv`, per section 9.1's example tree).
`persistence.py`'s `RunDirectory` has no method for this yet, and can't
until `runner.py` actually returns the captured output -- right now
`subprocess.run(capture_output=True)` captures it and then it's just
thrown away. One `runner.py` fix (return the captured output) unblocks
both this and the `error_message` issue above.

## Resolved

_(move fixed entries here with the PR number instead of deleting them)_
