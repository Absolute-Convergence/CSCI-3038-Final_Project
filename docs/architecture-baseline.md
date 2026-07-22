# Black Box Optimizer Architecture Baseline

## Purpose

Black Box Optimizer is a local Python 3.13.14 application. It will optimize an
external worker without importing or inspecting the worker's internal code.
The worker receives candidate parameters through command-line flags and writes
metrics to a trial-specific, one-row CSV file.

This document records the approved foundation contracts. It does not authorize
features beyond the MVP.

## MVP Boundaries

- Execution is local, synchronous, and sequential.
- No more than one worker trial may run at a time.
- Seeded RandomSearch is the only required search implementation.
- Every optimization problem has at least two objectives.
- Each objective independently declares `minimize` or `maximize`.
- Every attempted execution will eventually produce one immutable trial record.
- The final result will be the complete non-dominated Pareto Front, not a
  weighted winner.
- Networking, concurrency, databases, resume behavior, and worker inspection
  are excluded.
- A GUI may present results later but cannot own optimizer behavior.
- PyTorch may belong to the Iris example worker but is not an optimizer
  dependency.

## Foundation Domain Contracts

The initial model layer introduces only configuration and candidate types:

- `ParameterKind`: `integer`, `float`, or `categorical`.
- `ParameterDefinition`: a parameter name, kind, numeric bounds, or categorical
  choices.
- `Direction`: `minimize` or `maximize`.
- `Objective`: an exact worker metric name and its direction.
- `OptimizationContract`: ordered parameter definitions and objectives.
- `WorkerSpec`: worker command, metrics-output argument, and timeout.
- `AlgorithmSpec`: algorithm name and random seed.
- `StopPolicy`: maximum attempted trials.
- `ProjectConfiguration`: the four top-level configuration sections.
- `CandidateConfiguration`: one complete parameter mapping copied into a
  read-only view.

Frozen dataclasses and tuples protect the public contracts. Constructors copy
incoming sequences and candidate mappings so callers cannot retain mutable
aliases. Candidate legality against a particular `OptimizationContract` will
be implemented at a later boundary.

## External JSON Shape

The example configuration uses these top-level keys:

```text
worker
optimization
algorithm
stop_policy
```

Parameter definitions use `kind`. Algorithm configuration uses `name` and
`seed`. Parameter order and objective order in JSON are significant and must be
preserved by the future configuration loader.

## Iris Demonstration Contract

The planned external Iris worker accepts:

- `learning_rate`
- `hidden_size`
- `epochs`
- `batch_size`

It returns:

- `validation_accuracy` (`maximize`)
- `validation_loss` (`minimize`)
- `training_time_seconds` (`minimize`)

The current example is configuration only. No worker, PyTorch dependency,
search algorithm, subprocess runner, history, or Pareto evaluator is part of
this foundation patch.

