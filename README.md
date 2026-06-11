### (Heavily experimental. Still in development)


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

### EVB Setup From Cluster-Model IRC

A cluster-model IRC multi-XYZ is treated as geometry/path information only. XYZ comments may contain labels such as `RC`, `TS`, or `PROD`, but the code does not read thermodynamic reference energies from comments. Supply reference free energies separately, usually relative cluster-model values with frequency corrections:

```yaml
irc:
  path: inputs/hg317_irc.xyz
  order: prod_ts_rc   # original file is PROD -> TS -> RC; canonicalized to RC -> TS -> PROD
  rc_frame: auto      # explicit frame indices are original 0-based indices
  ts_frame: auto
  product_frame: auto

reference_profile:
  units: kcal/mol
  rc: 0.0
  ts: 18.5
  product: 7.2
  source_label: "CM g-xTB + frequency correction"

analysis:
  barrier:
    derive_regions_from_irc: true

sampling:
  mode: gap_umbrella
  windows:
    gap_umbrella:
      from_irc_scan: true
      n_windows: 41
      force_constant_kj_mol2: 0.01
```

Run the setup before expensive sampling:

```bash
evb setup-from-irc --config config.yaml --write-window-config
```

This writes pre-fit and post-fit diabatic scans, fits `delta_alpha` and constant `H12` from canonical RC/TS/PROD frames, reports warnings, and proposes gap umbrella centers/seeds. The final activation free energy must still come from EVB sampling plus WHAM/MBAR-style reconstruction over the energy-gap coordinate with good window overlap. The IRC is not an EVB PMF.

For cluster-model IRCs, enable `irc.relaxation` to generate locally relaxed full-system seeds before fitting/window generation. The setup first minimizes solvent, then the local protein/active-site region with alpha carbons and the far field restrained, then uses that pre-relaxed structure as the base for short IRC-frame minimizations. Set `irc.relaxation.platform: CUDA` with `require_platform: true` for production so these relaxations do not silently run on CPU.

Before using ordinary AMBER endpoint `.prmtop` files for EVB, run `evb prepare-evb-inputs --config config.yaml --output prep/system/evb_ready`. This writes EVB-ready `system.xml` bundles with consistent reactive nonbonded exclusions across states and a derived YAML config that points to them. Refit EVB parameters after this step.

Run conventional sampling first. Mapped windows and gap umbrellas are native OpenMM workflows and do not depend on PLUMED. Analysis writes `analysis/window_overlap.json`, `analysis/pmf_gap.csv`, and `analysis/barrier_estimate.json`. The current histogram mode is diagnostic; treat barriers as trustworthy only after overlap, blocking, and replicate checks.

PLUMED support is optional and currently intended for geometrical CVs such as distances, distance differences, and 2D bond formation/breaking CVs. PLUMED atom indices are 1-based. OpenMM/Python indices are 0-based. Energy-gap OPES is not automatically supported because PLUMED cannot see the internal EVB gap without an explicit bridge.

## Examples

- `examples/toy_evb.yaml`: fast OpenMM-bundle template for tests and development
- `examples/solution_template.yaml`: solution-reaction skeleton
- `examples/enzyme_template.yaml`: enzyme-reaction skeleton
- `examples/plumed_opes_template.yaml`: geometrical OPES template with restart/COLVAR files

Read `docs/validation.md` before trusting any EVB barrier.

Native tabulated EVB gap biasing for `evb-gap-table-metad` is documented in [`docs/native_gap_bias.md`](docs/native_gap_bias.md). This path avoids the separate OpenMM `app.Metadynamics`/`BiasVariable` gap-CV evaluation, but it does not remove duplicated full PME in exact decomposed systems.

For the HG3.17 IRC-seeded gap-metadynamics example and 2-GPU launcher, see
`docs/hg317_reproducibility.md`. Large generated outputs are intentionally not
tracked in git; regenerate EVB-ready bundles, trajectories, PMFs, and
visualization pseudo-trajectories locally.
