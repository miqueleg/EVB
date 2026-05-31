# Kemp EVB OpenMM PoC

Proof-of-concept codebase for conventional two-state EVB in OpenMM, organized around:

1. compatible diabatic state definitions
2. native OpenMM EVB and mapped Hamiltonians
3. mapping-window sampling
4. gap-distribution analysis and barrier reconstruction
5. a later path to enhanced sampling such as OPES

The code is being built in phases. The repository now includes:

- a native lower-surface EVB propagator in OpenMM
- a mapped-Hamiltonian sampling path for conventional EVB windows
- per-frame logging of `E1`, `E2`, `ΔE`, shifted `ΔE`, EVB energy, and diagnostic distances
- basic analysis for gap histograms, PMF-like reconstruction, window overlap, and barrier summaries
- bootstrap and ensemble EVB parameter fitting

## Current Scope

Implemented now:

- two-state EVB with constant `H12`
- one offset parameter `delta_alpha`
- one native OpenMM `System` for the lower EVB surface
- one mapped OpenMM `System` for windowed sampling:
  - `E_map(lambda) = (1 - lambda) * E1 + lambda * (E2 + delta_alpha)`
- JSON and YAML config loading
- state compatibility validation
- single-point EVB and gap evaluation
- mapping-window sampling and series execution
- gap histogramming, PMF export, overlap diagnostics, and barrier estimation
- bootstrap fitting from explicit calibration targets
- ensemble fitting by scanning `delta_alpha` and `H12` against sampled window data

Not implemented yet:

- rigorous reweighting / WHAM / MBAR-style reconstruction
- OPES integration
- geometry-dependent coupling
- production chemistry preparation workflows

## Repository Layout

```text
src/kemp_evb/
  analysis/      gap histograms, PMF reconstruction, overlap, barrier summaries
  engine/        validation helpers for diabatic states
  observables/   gap and geometric diagnostics
  sampling/      mapping windows and sampling runners
  cli.py         command-line entry points
  config.py      legacy JSON + new YAML config models
  evb.py         EVB Hamiltonian and bootstrap fitting logic
  openmm_backend.py
  simulation.py

examples/
  kemp_evb_config.json
  solution_kemp_baseline.yaml

tests/
  test_evb_core.py
  test_openmm_toy.py
  test_config_modern.py
  test_gap_observables.py
  test_sampling_windows.py
  test_analysis_pipeline.py
  test_fitting_pipeline.py
```

## Installation

Minimal editable install:

```bash
pip install -e .
```

Recommended dependencies for the current code:

```bash
pip install openmm PyYAML pytest
```

If you use the included conda environment file:

```bash
conda env create -f environment.kemp-evb.yml
conda activate kemp-evb
pip install -e .
```

## Input Requirements

For real EVB runs, the code expects two diabatic states with identical system layout:

- same number of atoms
- same atom ordering
- same masses
- same periodic box
- same constraints / virtual-site structure

Current production input format:

- `state1.prmtop`
- `state1.inpcrd`
- `state2.prmtop`
- `state2.inpcrd`

The repository does not currently prepare these states for you. Chemistry preparation is assumed to happen outside this package.

## Config Formats

Two config styles are supported.

### Legacy JSON

The original prototype JSON format still works:

- [kemp_evb_config.json](/home/mestevez/Projects/EVB/PoC1/KEMP/examples/kemp_evb_config.json)

This is mainly kept for backward compatibility with the old `calibrate`, `singlepoint`, `minimize`, and `md` commands.

### Baseline YAML

The new workflow-oriented config is:

- [solution_kemp_baseline.yaml](/home/mestevez/Projects/EVB/PoC1/KEMP/examples/solution_kemp_baseline.yaml)

This format adds:

- project metadata
- reaction metadata and reactive atom indices
- EVB coupling parameters
- mapping-window definitions
- analysis settings
- diagnostic distance definitions

## CLI

### Legacy EVB surface commands

```bash
kemp-evb calibrate --config examples/kemp_evb_config.json
kemp-evb singlepoint --config examples/kemp_evb_config.json
kemp-evb minimize --config examples/kemp_evb_config.json
kemp-evb md --config examples/kemp_evb_config.json
```

These operate on the lower EVB surface using the native OpenMM EVB system.

### Phase 1 validation and observables

Validate state compatibility:

```bash
kemp-evb validate-states --config examples/solution_kemp_baseline.yaml
```

Evaluate gap observables on the configured coordinates:

```bash
kemp-evb gap-eval --config examples/solution_kemp_baseline.yaml
```

Evaluate on a different structure:

```bash
kemp-evb gap-eval --config examples/solution_kemp_baseline.yaml --coords some_structure.pdb
```

### Phase 2 sampling

Run one mapping window:

```bash
kemp-evb sample-window --config examples/solution_kemp_baseline.yaml --window w000
```

Run the whole mapping series:

```bash
kemp-evb sample-series --config examples/solution_kemp_baseline.yaml
```

### Phase 3 analysis

Analyze gap distributions and write analysis artifacts:

```bash
kemp-evb analyze-gap --config examples/solution_kemp_baseline.yaml
```

Rebuild the barrier estimate from existing window logs:

```bash
kemp-evb reconstruct-barrier --config examples/solution_kemp_baseline.yaml
```

Write a compact summary report:

```bash
kemp-evb report --config examples/solution_kemp_baseline.yaml
```

### Phase 4 fitting

Bootstrap fit from explicit calibration energies:

```bash
kemp-evb fit-bootstrap --config examples/kemp_evb_config.json
```

Ensemble fit from existing mapping-window logs:

```bash
kemp-evb fit-ensemble --config examples/solution_kemp_baseline.yaml
```

## Output Layout

With the new YAML workflow, outputs are organized as:

```text
outputs/<project>/
  state_validation.json
  gap_eval.json

  windows/
    w000/
      window_spec.json
      production_observables.csv
      final_state.pdb
      summary.json
    w001/
      ...

  series/
    window_index.json

  analysis/
    analysis_report.json
    gap_histograms.json
    pmf_gap.csv
    window_overlap.json
    barrier_estimate.json

  fitting/
    bootstrap_fit.json
    ensemble_fit.json
    fit_scan.csv

  reports/
    summary.json
    summary.md
```

The key artifact for conventional EVB analysis is:

`windows/wXXX/production_observables.csv`

Each row contains:

- frame index
- MD step
- time
- window id
- mapping `lambda`
- `E1`
- `E2`
- raw energy gap
- shifted energy gap
- EVB lower-surface energy evaluated from the current parameters
- EVB weights
- optional geometric diagnostics

## Analysis Model

The current Phase 3 analysis is intentionally simple.

It does:

1. load all per-window `ΔE` values
2. build per-window histograms
3. build one combined histogram
4. convert the combined probability distribution into a PMF-like curve:
   - `F(ΔE) = -RT ln P(ΔE) + C`
5. estimate:
   - reactant basin minimum
   - product basin minimum
   - a barrier maximum between them
6. compute a pairwise overlap matrix between window histograms

This is useful as a baseline and debugging tool. It is not yet a rigorous multi-window free-energy reconstruction method.

## Fitting Model

Two fitting paths exist.

### Bootstrap fit

This uses explicit calibration energies already present in the legacy config:

- MM energies for `min1`, `min2`, and `TS`
- target QM/MM or reference energies for the same three points

It is a direct pointwise initialization method.

### Ensemble fit

This uses sampled window logs and scans over:

- `delta_alpha`
- `H12`

For each candidate pair, the code:

1. reads `E1`, `E2`, and `lambda` from the window logs
2. computes candidate lower-surface EVB energies
3. applies a lightweight reweighting from the mapped Hamiltonian used during sampling
4. reconstructs a PMF-like profile
5. scores the candidate against:
   - target forward barrier
   - target reaction free energy

The current implementation is intentionally simple and is meant as an operational baseline, not a final statistically rigorous fitting engine.

## Tests

Run tests in the intended environment:

```bash
python -m pytest -q
```

Current coverage includes:

- analytical EVB core behavior
- native OpenMM EVB toy systems
- YAML config loading
- gap observables
- mapping-window generation and logging
- analysis pipeline on synthetic window data
- fitting pipeline on synthetic window data

## Recommended Workflow

For a real solution benchmark:

1. prepare `state1` and `state2` externally
2. fill in `examples/solution_kemp_baseline.yaml`
3. run `validate-states`
4. run `gap-eval`
5. run one or a few `sample-window` jobs to verify stability
6. run `sample-series`
7. run `analyze-gap`
8. optionally run `fit-ensemble`
9. inspect:
   - `analysis/pmf_gap.csv`
   - `analysis/window_overlap.json`
   - `analysis/barrier_estimate.json`
   - `fitting/ensemble_fit.json`
   - `fitting/fit_scan.csv`

Only after the baseline is numerically sane should enhanced sampling be added.

## Status Relative To The Long-Term Goal

This repo is now aligned with a conventional EVB baseline:

- the primary observable is the diabatic energy gap
- mapped windows are sampled explicitly
- barrier reconstruction is statistical, based on sampled gap distributions

The next major steps are:

1. stronger reweighting / free-energy reconstruction
2. better uncertainty estimates for barriers and fits
3. OPES as an optional accelerated sampling backend

That order matters. OPES should be compared against a working conventional EVB baseline, not used to compensate for an unvalidated baseline.
