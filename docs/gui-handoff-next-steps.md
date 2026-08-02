# Hyperloop GUI Handoff: Recommended Next Steps

- Status: implementation guidance, not a contract amendment
- Date: 2026-08-01
- GUI owner: Charles
- Architecture and integration owner/reviewer: Mel
- Reviewed GUI source: Charles's `PythonGUI` repository at commit
  [`2e56c8d`](https://github.com/Clfreeman22/PythonGUI/tree/2e56c8dd99f4516bc4e8d4cc78ed9ee79448a4b9)

## Purpose

This document recommends the next implementation steps for connecting
Charles's optional GUI to the completed optimizer. The product name is
**Hyperloop**. The existing Python package and module command remain
`black_box_optimizer` until a separately approved compatibility change is
made.

This guide does not change the approved JSON, CLI, CSV, result, reporting, or
persistence contracts. It deliberately uses the artifacts and command already
implemented on `main`.

## Architectural Boundary

The GUI is a presentation client of Hyperloop. It may collect configuration,
launch Hyperloop, and present completed output, but it must not:

- launch the configured worker itself;
- select or modify candidate values after configuration is submitted;
- create or rewrite trial history;
- recalculate the authoritative Pareto Front;
- choose a weighted winner; or
- become a dependency of the optimizer package.

Reporter continues to own durable result exports and Matplotlib rendering.
The GUI consumes those outputs without changing their meaning.

```text
Tkinter GUI
    |
    | writes a temporary project configuration and launches
    v
python -m black_box_optimizer CONFIG --output-dir GUI_JOB_DIRECTORY
    |
    | creates exactly one unique run directory
    v
run_*/
    |-- resolved_config.json
    |-- history.csv
    |-- pareto_front.csv
    |-- summary.txt
    |-- pareto_front.png
    `-- trials/
            |
            `----> GUI reads and presents; it never rewrites
```

The GUI's background thread does not authorize worker concurrency. It waits
for one Hyperloop process while the Tkinter event loop remains responsive.
Hyperloop's controller and runner remain responsible for ensuring that no more
than one worker trial is active.

## Findings from Charles's Current GUI

The reviewed GUI is a single Tkinter application. It already uses Pillow and
`ImageTk`, writes a temporary JSON file, launches a subprocess from a
background thread, and returns UI notifications to the Tkinter thread through
`root.after`. Those choices support a small, clean first integration.

The following defects should be corrected before connecting the result view:

1. `build_config_dict()` returns from inside the objectives loop. A
   multi-objective configuration therefore contains only its first objective,
   and an empty objective list returns no configuration. Move the return after
   the complete loop and validate the completed document before launch.
2. `_run_worker()` currently builds its command from
   `config["worker"]["command"]` and appends the configuration path. That
   invokes the opaque worker directly. The GUI must invoke Hyperloop; the
   worker command remains data inside the JSON configuration for Hyperloop's
   runner to use.
3. The subprocess call uses `worker.timeout_seconds` as the timeout for the
   entire optimization. That value is a per-trial worker timeout. A valid run
   may take roughly that duration for each authorized trial, plus application
   overhead. The GUI should not impose that value on the overall run.
4. The GUI reports process completion but does not retain an unambiguous run
   directory or load any final artifacts.

The relevant reviewed code is in
[`Hyperloop.py`](https://github.com/Clfreeman22/PythonGUI/blob/2e56c8dd99f4516bc4e8d4cc78ed9ee79448a4b9/Hyperloop.py),
especially `build_config_dict`, `run_loop`, and `_run_worker`.

## Recommended MVP Handoff

### 1. Keep Configuration Assembly Pure and Complete

Extract configuration assembly and value conversion from widget manipulation
where practical. Build every declared parameter and objective before returning
the document. Preserve their UI order because Hyperloop preserves parameter
and objective order as part of the configuration contract.

Before starting a background thread, the GUI should reject incomplete names,
invalid bounds, invalid categorical choices, and unparseable numeric fields
with an actionable message. Hyperloop remains the final authority and will run
its complete loader validation again.

Human collaboration: Charles owns these GUI-only corrections. Mel's approval
is required if the proposed fix changes the accepted JSON shape rather than
merely producing the existing shape correctly.

### 2. Launch Hyperloop, Not the Worker

Run the GUI itself under the required Python 3.13.14 environment. Construct the
optimizer command as an argument list:

```python
command = [
    sys.executable,
    "-m",
    "black_box_optimizer",
    str(configuration_path),
    "--output-dir",
    str(gui_job_directory),
]
```

Use `shell=False`, capture stdout and stderr, and perform the blocking wait
only on the GUI's background thread. Continue to schedule every Tkinter widget
update through `root.after`; Tkinter widgets must not be updated directly from
the background thread.

During development from separate repositories, the subprocess working
directory must be the Hyperloop checkout unless the package has otherwise been
made importable in the Python 3.13 environment. Final integration should not
depend on a hard-coded path to either developer's checkout.

Do not use `worker.timeout_seconds` as an overall subprocess timeout. If a GUI
cancel button is later required, coordinate its process signaling and
partial-result behavior with Mel before implementing it. Forcefully ending the
Hyperloop parent process could leave its active worker without a chance to
record cancellation evidence.

Human collaboration: Charles owns command construction and GUI threading. Mel
must review any requested change to Hyperloop's CLI or cancellation contract.

### 3. Give Each GUI Invocation an Unambiguous Output Base

The GUI should start each launch with a newly created, empty job directory
under a user-selected or application-owned output root. A UUID is sufficient:

```text
chosen_output_root/
`-- gui_job_<uuid>/
    `-- run_<timestamp>_<token>/
```

Pass `gui_job_<uuid>` as `--output-dir`. After the process exits, the GUI can
require exactly one `run_*` child. This avoids scraping human-readable stdout
and uses the current CLI contract unchanged.

If the GUI is integrated into this repository, the default may live below the
repository-root `runs/` directory, which Git already ignores. If Charles runs
the GUI from its separate repository, use a user-selected directory or an
operating-system application-data location rather than writing generated CSV,
PNG, and trial files into the GUI source tree.

Handle output discovery explicitly:

- no `run_*` directory means initialization failed before durable run creation;
- one `run_*` directory is the expected handoff;
- more than one indicates that the job directory was reused or corrupted and
  must not be guessed around; and
- discovered output must be preserved on failed and cancelled runs because it
  may contain valid partial evidence and a partial Pareto Front.

### 4. Use Existing Files for the First Result View

The current output set already covers the first GUI result screen:

| Artifact | GUI use | Interpretation rule |
| --- | --- | --- |
| `summary.txt` | Show the completion summary verbatim | Do not infer a universal winner |
| `pareto_front.png` | Display the default plot through Pillow | Explanatory view of the first two objectives |
| `pareto_front.csv` | Populate a Pareto-results table | Preserve every row and declared column order |
| `history.csv` | Populate optional trial/history details | Include unsuccessful and ineligible attempts |
| `resolved_config.json` | Show the configuration actually used | Treat it as read-only |
| `trials/*/stdout.txt` and `stderr.txt` | Optional diagnostics view | Do not replace bounded user-facing summaries |

For the plot, open `pareto_front.png` with `PIL.Image`, resize only a display
copy while preserving aspect ratio, and retain the corresponding
`ImageTk.PhotoImage` reference for the lifetime of the widget. Charles's GUI
already declares Pillow, so this adds no dependency.

The result screen should show the complete Pareto set. It may let a user select
a row or compare trials, but it must not highlight one row as Hyperloop's
automatic winner.

### 5. Interpret Completion Without Losing Evidence

Use the implemented CLI exit codes:

- `0`: normal completion or no eligible trials;
- `1`: fatal run or reporting failure;
- `2`: invalid initialization or configuration; and
- `130`: user cancellation.

When a run directory exists, display its summary and available artifacts even
after a nonzero exit. A failed or cancelled result may still expose a
partial-but-valid Pareto Front. When no run directory exists, show the bounded
CLI error and keep the user's unsaved form state available for correction.

Do not parse worker stdout as optimizer results. Complete worker streams belong
to trial-local diagnostic files, and metrics belong to the persisted CSV
artifacts.

### 6. Add Focused GUI Verification

Charles's GUI handoff should include automated coverage for at least:

- two or more objectives surviving configuration construction in order;
- integer, float, and categorical parameter conversion;
- exact Hyperloop command construction with an argument list and
  `shell=False`;
- use of a unique, empty GUI job directory;
- zero, one, and multiple discovered `run_*` directories;
- exit-code-to-message behavior;
- preservation and presentation of artifacts after failure or cancellation;
- loading the PNG through Pillow; and
- scheduling UI updates through `root.after` rather than touching widgets
  from the worker thread.

The acceptance handoff should then run the real Iris configuration through the
GUI under Python 3.13.14 and confirm:

1. all three declared Iris objectives reach `resolved_config.json`;
2. Hyperloop, rather than the Iris worker, is the process the GUI launches;
3. no more than one Iris worker is active at a time;
4. the result screen shows the summary, complete Pareto table, and PNG;
5. generated artifacts do not appear in either Git worktree; and
6. the same run remains usable from the CLI without the GUI installed.

Human collaboration: Charles should demonstrate and review GUI behavior. Mel
should review the final cross-repository integration, result interpretation,
and proof that the optimizer remains independently runnable.

## Why the Handoff Should Not Use Pickle

A pickled Matplotlib `Figure` or pickled optimizer result is not recommended:

- the GUI already has Pillow and can display the durable PNG directly;
- CSV and JSON artifacts are inspectable and usable outside one Python process;
- pickles are tied to Python object and library versions;
- loading a pickle is unsafe when its provenance or integrity is uncertain;
  and
- a serialized figure would blur ownership between Reporter and the GUI.

If interactive plots become a real requirement, propose that separately. A
future GUI can read the existing CSV/JSON data and construct a Tkinter-embedded
Matplotlib figure. That would add a GUI dependency and presentation behavior,
but it still would not make a pickled figure authoritative.

## Naming Guidance

Use **Hyperloop** in window titles, user-facing text, and new documentation.
Use lowercase `hyperloop` for a future executable or console-script name. The
current technical module command remains:

```text
python -m black_box_optimizer
```

Renaming the Python package, removing that module command, or adding a new
`hyperloop` console entry point changes public installation or CLI behavior.
Those changes require Mel's approval and a compatibility plan. A branding-only
documentation cleanup can proceed separately without pretending the technical
namespace has already changed.

## Approval and Collaboration Gates

| Proposed action | Required owner or approval |
| --- | --- |
| Correct GUI assembly of the existing JSON shape | Charles |
| Launch the existing CLI and read existing artifacts | Charles, with Mel reviewing integration |
| Present PNG/CSV/JSON with existing Tkinter and Pillow dependencies | Charles |
| Change JSON, CLI, CSV, result, reporting, or persistence contracts | Mel before implementation |
| Add `result.json`, `--json`, `--run-dir`, or another machine interface | Mel before implementation |
| Add a `hyperloop` executable or rename `black_box_optimizer` | Mel before implementation |
| Add Matplotlib to the GUI or create an interactive plot contract | Mel dependency approval and Charles ownership |
| Add any pickle output | Mel approval; current recommendation is not to add it |
| Define cross-process cancellation behavior | Mel and Charles together |

## Recommended Checkpoint Order

Keep the GUI work reviewable in small checkpoints:

1. **Configuration correctness:** fix the objective-loop return and add pure
   configuration tests.
2. **Launch adapter:** invoke the existing Hyperloop CLI with an argument list,
   no overall worker-timeout reuse, and a unique job directory.
3. **Artifact loader:** locate exactly one run and load its existing outputs
   read-only.
4. **Result view:** show summary, complete Pareto table, and PNG through
   Tkinter/Pillow.
5. **Failure handling:** preserve and display available partial evidence for
   failed and cancelled results.
6. **Integration acceptance:** run Iris end to end, verify repository hygiene,
   and hand the branch to Mel for review.
7. **Optional enhancements:** only after explicit approval, consider
   structured machine output, interactive plotting, packaging, a `hyperloop`
   console command, or coordinated cancellation.

Each checkpoint should state which repository changed, what Charles reviewed,
whether a controlled Hyperloop contract changed, and which verification was
run. GUI work should not be silently mixed into optimizer/controller changes.
