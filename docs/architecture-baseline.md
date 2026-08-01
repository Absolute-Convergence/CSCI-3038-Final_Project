# Black Box Optimizer Architecture Baseline

## Purpose

Black Box Optimizer is a local Python 3.13.14 application. It will optimize an
external worker without importing or inspecting the worker's internal code.
The worker receives candidate parameters through command-line flags and writes
metrics to a trial-specific, one-row CSV file.

This document records the approved foundation contracts. It does not authorize
features beyond the MVP.

This is an architectural authority document, not an implementation-status
tracker. `README.md` describes what exists on `main`, and the scratch memory in
`AGENTS.md` records the latest verified checkpoint and immediate next work.
The Draft v0.2 planning documents provide detailed guidance but remain pending
team ratification and technical approval; they do not silently override this
approved baseline or a later recorded human decision.

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

The foundation contract layer defines these configuration and candidate types:

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
aliases. `CandidateConfiguration` protects the mapping itself; legality against
a particular `OptimizationContract` belongs to a separate validation boundary
and is not stored as mutable candidate state.

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
preserved by the configuration loader.

## Iris Demonstration Contract

The external Iris worker accepts:

- `learning_rate`
- `hidden_size`
- `epochs`
- `batch_size`

It returns:

- `validation_accuracy` (`maximize`)
- `validation_loss` (`minimize`)
- `training_time_seconds` (`minimize`)

The worker remains outside `black_box_optimizer`. PyTorch may support this
example and its tests but must not be imported by, or become a runtime
dependency of, the optimizer package. Whether each component is currently
implemented belongs in `README.md`, not in this stable contract document.
