# Controller Integration Ownership and Initialization Direction

- Status: Approved
- Date: 2026-08-01
- Approver and reviewer: Mel
- Implementer and integration owner: Codex
- Affected areas: application composition, controller lifecycle, candidate
  validation, final results, reporting, cancellation, and diagnostics

## Context

Draft v0.2 models `INITIALIZING` as a controller state. The merged
`ApplicationController` instead receives constructed collaborators and begins
with candidate selection. Implementation work showed that configuration I/O,
run-directory creation, algorithm construction, stop-policy construction, and
Reporter construction have their own dependencies and failure behavior. They
are application composition concerns rather than transitions in the worker
attempt lifecycle.

## Decision

Initialization remains outside the lifecycle finite-state machine. A separate
application/composition boundary will:

1. load and validate the project configuration;
2. create the run directory and write the resolved configuration;
3. construct the search, stop-policy, and reporting collaborators; and
4. construct and invoke `ApplicationController`.

The controller begins in `SELECTING` only after that work succeeds. It owns the
sequential select, gate, execute, record, evaluate, fail, finalize, and stop
transitions. This is an approved emergent change in direction from the draft
TDS, not an accidental omission of `INITIALIZING`.

The controller integration also uses these approved policies:

- validate every proposal against the declared parameter domain before launch;
- treat a domain-invalid proposal as an algorithm defect, launch no worker,
  and finalize with a fatal result;
- treat a domain-valid worker failure as one recoverable recorded trial;
- add no parameter-exclusion grammar to the MVP;
- expose any partial-but-valid Pareto front on cancelled or failed results;
- keep Reporter as a separate collaborator called by the controller;
- terminate an interrupted child, wait two seconds, then kill if necessary;
  and record exactly one cancelled trial; and
- keep `TrialRecord.error_message` at or below 1,000 characters using the last
  nonblank stderr line plus exit-code context, while preserving complete UTF-8
  replacement-decoded stdout and stderr in trial-local files.

## Consequences

- `ControllerState` intentionally has no `INITIALIZING` member.
- Initialization failures occur before controller construction and do not
  fabricate a `TrialRecord`, because no worker attempt was authorized.
- The application layer may depend on configuration, persistence, search,
  stop-policy, controller, and reporting modules. Those modules must not depend
  back on the application layer.
- Result and reporting contracts remain immutable and are subject to Mel's
  review before the controller/result work is accepted.
- This record explains the TDS direction change without a broad rewrite of the
  governance documents.
