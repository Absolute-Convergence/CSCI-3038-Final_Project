# Package Release Bug-Remediation Decisions

Date: 2026-08-03

Mel approved these contract choices before the first PyPI publication:

1. A worker command containing an absolute path written for a different
   operating-system path syntax is rejected during configuration loading with
   a location-aware `ConfigurationError`. Hyperloop does not rewrite or defer
   an unusable foreign path to a later worker-launch failure.
2. An authorized attempt lost because an optimizer collaborator unexpectedly
   fails is preserved as an immutable `TrialRecord` with
   `execution_status="internal_error"`. The run then finalizes with
   `fatal_error`; the attempt is never silently dropped or masked by a retry.
3. Run evidence and report files are atomically replaced one file at a time.
   The final report collection is not an all-or-nothing transaction. A
   reporting failure produces a nonzero outcome and means the collection is
   incomplete, even though files already committed remain individually valid.

These choices preserve the sequential worker boundary, immutable evidence,
and current root-level artifact paths. They add no networking, concurrency,
resume behavior, database, or weighted-winner behavior.
