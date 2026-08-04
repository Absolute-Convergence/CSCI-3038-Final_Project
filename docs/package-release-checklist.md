# Hyperloop v0.1.1 Package Release Checklist

Status: package preparation in progress; nothing in this checklist authorizes
publication by itself.

## Release Identity

- PyPI distribution: `hyperloop-optimizer`
- Version: `0.1.1`
- Installed optimizer command: `hyperloop-optimizer`
- Installed synthetic-worker command: `hyperloop-synthetic-worker`
- Existing module command: `python -m black_box_optimizer`
- Core import package: `black_box_optimizer`
- Bundled worker package: `hyperloop_workers`

The first PyPI publication should be v0.1.1. GitHub v0.1.0 already identifies
the earlier commit `f2a4cbc39c595651998605d757f785bec5832931`; publishing the
new NSGA-II code to PyPI under that same version would make v0.1.0 identify two
different source states.

## Distribution Boundary

The wheel includes only:

- `black_box_optimizer`
- `hyperloop_workers.synthetic_worker`
- package metadata and the MIT license

The wheel excludes the GUI, Iris example, PyTorch, tests, planning documents,
and generated run evidence. The synthetic worker stays outside the optimizer
namespace, so Hyperloop continues to launch it as an opaque subprocess.

Direct dependencies are deliberately separated:

- `requirements.txt`: NumPy and Matplotlib for Hyperloop core/reporting
- `requirements-iris.txt`: PyTorch for the external Iris worker
- `requirements-gui.txt`: Pillow for the optional source-checkout GUI
- `requirements-release.txt`: build and upload tools

Matplotlib itself may install Pillow transitively for image output. Hyperloop
does not declare Pillow as a GUI dependency in its package metadata.

## Audit Evidence

- Updated `main` baseline: 366 tests passed after PR #24 fixed negative-seed
  validation and before that change was merged into the package branch.
- Remediated package tree: all 394 tests pass on Windows under Python 3.13.14,
  including the formerly failing invalid-path regression.
- Known-issue audit: PR #24 correctly fixes the negative-seed CLI crash. The
  other nine upstream reports and the Windows-specific finding are resolved on
  the package branch with focused regression coverage. `KNOWN_ISSUES.md` has
  no confirmed open defect at this checkpoint.
- Generated-output barrier: `runs/`, `optimizer_runs/`, and ZDT1 comparison
  results are ignored.
- Repeated-run test: an older two-trial Pareto front cannot enter a later
  one-trial run or overwrite its reports.
- Post-remediation ZDT1 comparison: 10 seeds and 500 trials per algorithm,
  totaling 10,000 real sequential worker subprocesses in 687.6 seconds.
  - Random Search mean final hypervolume: 0.2113 (24.1% of optimum), standard
    deviation 0.1119.
  - NSGA-II mean final hypervolume: 0.3884 (44.3% of optimum), standard
    deviation 0.0924.
  - NSGA-II produced about 84% more dominated hypervolume than Random Search
    on average. Raw traces contain exactly 5,000 trials per algorithm; the CSV
    and convergence chart remain ignored generated evidence.
- Remediated-tree build: the wheel and source archive build successfully and
  pass `twine check`. The wheel contains 30 entries, excludes the GUI, Iris,
  examples, tests, documentation, and PyTorch, and exposes both approved
  console commands.
- Clean install: direct requirements are only NumPy and Matplotlib; PyTorch is
  absent; both `hyperloop-optimizer --help` and
  `hyperloop-synthetic-worker --help` succeed outside the checkout.
- Installed smoke run: four valid synthetic trials completed and produced a
  three-member Pareto front with all required report artifacts. This checks
  package wiring only; it is separate from the 10,000-trial efficacy result.
- Core-only test environment: 380 tests pass and four Iris/PyTorch modules are
  skipped as intended.

## Remaining Gates

- [x] Resolve the validation, CLI, cross-platform path, controller-state, and
  persistence defects recorded in `KNOWN_ISSUES.md`.
- [x] Decide whether result reporting must be atomic across the entire artifact
  group or whether the public persistence contract must explicitly promise
  only per-file atomicity.
- [x] Run the final complete suite with PyTorch installed.
- [x] Run the repository source-hygiene checker after the last change.
- [ ] Open a pull request and require the Windows, Ubuntu, macOS, and package
  jobs in `.github/workflows/package.yml` to pass.
- [ ] Build fresh archives from the exact release commit.
- [ ] Inspect archive contents and rerun `python -m twine check dist/*`.
- [ ] Upload to TestPyPI and install v0.1.1 into a fresh Python 3.13.14
  environment using only that index plus the public dependency index.
- [ ] Run the installed command and synthetic worker from outside the source
  checkout.
- [ ] Create the v0.1.1 Git tag and GitHub release from the verified commit.
- [ ] Upload the exact TestPyPI-validated archives to production PyPI.
- [ ] Install `hyperloop-optimizer==0.1.1` from production PyPI and repeat the
  smoke test.
