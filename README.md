# OpenMM EVB

Reusable two-state empirical valence bond (EVB) tooling for OpenMM. The validated baseline is the conventional lower EVB surface

```text
E_EVB = 0.5*(E1 + E2 + delta_alpha) - sqrt(0.25*(E1 - E2 - delta_alpha)^2 + H12^2)
```

The central observable is the shifted diabatic energy gap:

```text
gap = E1 - E2 - delta_alpha
```

The old `kemp_evb` package and `kemp-evb` command remain as compatibility aliases. New code should prefer `evb` imports and the `evb` CLI.

## Scope

Implemented:

- constant-`H12` two-state EVB Hamiltonian in kJ/mol
- native OpenMM lower-surface EVB system using `CustomCVForce`
- mapped Hamiltonian windows: `E_map(lambda) = (1-lambda)*E1 + lambda*(E2 + delta_alpha)`
- native OpenMM gap umbrellas: `0.5*k*(gap-gap0)^2`
- AMBER `prmtop/inpcrd` and serialized OpenMM `system.xml` plus PDB coordinates
- state compatibility checks for atom count/order, masses, constraints, virtual sites, and boxes
- diagnostic histogram PMFs, overlap matrices, barrier summaries, bootstrap/ensemble fitting
- optional openmm-plumed attachment for geometrical CV biasing

Not yet validated:

- geometry-dependent `H12`
- automatic PLUMED access to the internal OpenMM EVB energy gap
- production-quality MBAR uncertainty for every workflow

## Install

```bash
pip install -e ".[openmm,test]"
```

For analysis and PLUMED:

```bash
pip install -e ".[openmm,analysis,plumed,test]"
```

Conda/mamba users can start from:

```bash
mamba env create -f environment.yml
mamba activate openmm-evb
```

## Inputs

Prepare two diabatic states with identical atom layout. The code validates:

- same atom count and atom names/order
- same particle masses
- same constraints and virtual-site structure
- compatible periodic boxes

Supported formats:

- AMBER: `topology: state.prmtop`, `coordinates: state.inpcrd`
- OpenMM bundle: `format: openmm`, `topology: system.xml`, `coordinates: coordinates.pdb`

All package APIs use kJ/mol, nm, and ps.

## CLI

```bash
evb validate --config examples/toy_evb.yaml
evb singlepoint --config examples/toy_evb.yaml
evb minimize --config examples/toy_evb.yaml
evb run-md --config examples/toy_evb.yaml
evb sample-window --config examples/toy_evb.yaml --window w000
evb sample-series --config examples/toy_evb.yaml
evb analyze --config examples/toy_evb.yaml
evb fit --config examples/toy_evb.yaml
evb plumed-md --config examples/plumed_opes_template.yaml
evb make-template --kind toy
```

## Workflows

Fit `delta_alpha` and `H12` from calibration targets with `evb fit` or from explicit reference energies with the legacy bootstrap command. Keep `H12` constant unless you add and test a new coupling API.

Run conventional sampling first. Mapped windows and gap umbrellas are native OpenMM workflows and do not depend on PLUMED. Analysis writes `analysis/window_overlap.json`, `analysis/pmf_gap.csv`, and `analysis/barrier_estimate.json`. The current histogram mode is diagnostic; treat barriers as trustworthy only after overlap, blocking, and replicate checks.

PLUMED support is optional and currently intended for geometrical CVs such as distances, distance differences, and 2D bond formation/breaking CVs. PLUMED atom indices are 1-based. OpenMM/Python indices are 0-based. Energy-gap OPES is not automatically supported because PLUMED cannot see the internal EVB gap without an explicit bridge.

## Examples

- `examples/toy_evb.yaml`: fast OpenMM-bundle template for tests and development
- `examples/solution_template.yaml`: solution-reaction skeleton
- `examples/enzyme_template.yaml`: enzyme-reaction skeleton
- `examples/plumed_opes_template.yaml`: geometrical OPES template with restart/COLVAR files

Read `docs/validation.md` before trusting any EVB barrier.
