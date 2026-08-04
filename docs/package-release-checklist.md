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

- Updated `main` baseline: 363 tests passed before the package branch was
  merged with the upstream bug-audit and NSGA-II crowding-distance fix.
- Merged package tree: 372 tests collected on Windows; 371 pass and one errors
  because an invalid configuration path leaks a raw `ValueError`. This new
  Windows-specific defect is recorded in `KNOWN_ISSUES.md`.
- Known-issue audit: all ten upstream open reports were reproduced on the
  package branch. Together with the Windows-specific finding, eleven confirmed
  defects remain open; no PyPI upload is approved while release blockers are
  unresolved.
- Generated-output barrier: `runs/`, `optimizer_runs/`, and ZDT1 comparison
  results are ignored.
- Repeated-run test: an older two-trial Pareto front cannot enter a later
  one-trial run or overwrite its reports.
- Full ZDT1 comparison: 5 seeds and 500 trials per algorithm, totaling 5,000
  real sequential worker subprocesses.
  - Random Search mean final hypervolume: 0.2370 (27.0% of optimum), standard
    deviation 0.1568.
  - NSGA-II mean final hypervolume: 0.3890 (44.4% of optimum), standard
    deviation 0.0692.
- Isolated build: wheel and source archive built successfully and passed
  `twine check`.
- Clean install: direct requirements were only NumPy and Matplotlib; PyTorch
  was absent; `hyperloop-optimizer --help` named the installed command.
- Installed smoke run: four valid synthetic trials completed and produced a
  three-member Pareto front with all required report artifacts.
- Core-only test environment: 277 tests passed and four Iris/PyTorch modules
  skipped as intended.

## Remaining Gates

- [ ] Resolve the validation, CLI, cross-platform path, controller-state, and
  persistence defects recorded in `KNOWN_ISSUES.md`.
- [ ] Decide whether result reporting must be atomic across the entire artifact
  group or whether the public persistence contract must explicitly promise
  only per-file atomicity.
- [ ] Run the final complete suite with PyTorch installed.
- [ ] Run the repository source-hygiene checker after the last change.
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
