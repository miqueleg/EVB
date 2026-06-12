# OpenMM EVB

A general two-state empirical valence bond (EVB) package for OpenMM. A user provides two diabatic states, a reference reaction profile or barrier, optional IRC/frame mapping, an EVB representation, and a sampling workflow. The package validates the inputs, calibrates EVB parameters, generates umbrella or native table-metadynamics configs, runs sampling, analyzes PMF/FEL barriers, and compares the result to the reference profile.

The lower EVB surface is

```text
E_EVB = 0.5*(E1 + E2 + delta_alpha) - sqrt(0.25*(E1 - E2 - delta_alpha)^2 + H12^2)
```

The central coordinate is the shifted diabatic gap:

```text
gap = E1 - E2 - delta_alpha
```

## EVB Representations

- `full_state`: exact legacy-style endpoint-state EVB using complete state1/state2 OpenMM systems.
- `q_region`: Q-region EVB, where common environment terms are evaluated once and only Q-region residual energies are mixed.

For Q-region nonbonded terms, exact direct-space modes and explicitly approximate local PME modes are separated. Approximate modes are never labelled exact.

## Generic Workflow

Create a template:

```bash
.venv/bin/evb make-template --kind generic_q_region_local_pme --output config.yaml
```

Validate it:

```bash
.venv/bin/evb validate-workflow-config --config config.yaml
```

Generate Q-region candidates:

```bash
.venv/bin/evb make-qregion-candidates --config config.yaml --output outputs/qregion_candidates
```

Calibrate to arbitrary reference energies from a frame-energy CSV:

```bash
.venv/bin/evb calibrate-profile \
  --reference examples/generic_reference_profile.yaml \
  --coords frame_energies.csv \
  --output outputs/profile_calibration
```

Derive umbrella windows and workflow configs:

```bash
.venv/bin/evb derive-windows --config config.yaml --output outputs/windows
.venv/bin/evb run-workflow --config config.yaml --output outputs/workflow --workflow all --mode pilot
```

Analyze outputs:

```bash
.venv/bin/evb analyze-umbrella --coords outputs/window_observables.csv --output outputs/umbrella_analysis
.venv/bin/evb analyze-metad --coords outputs/native_bias/bias_table.csv --profile 15 --output outputs/metad_analysis
.venv/bin/evb compare-profile --reference examples/generic_reference_profile.yaml --output outputs/comparison
```

## Reference Profiles

Reference profiles are method-agnostic. Method labels such as `DFT`, `experiment`, `QMMM`, or any user-defined label are metadata, not code paths. Supported energy units include `kJ/mol`, `kcal/mol`, `eV`, and `hartree`. See [`docs/reference_profile_schema.md`](docs/reference_profile_schema.md).

## Native Table WT-MetaD

`evb-gap-table-metad` and generic `run-metad` use native well-tempered metadynamics on the EVB gap with the bias stored as an OpenMM tabulated function inside the EVB force. This avoids OpenMM `app.Metadynamics`/`BiasVariable` duplication. See [`docs/native_table_metadynamics.md`](docs/native_table_metadynamics.md).

## Documentation

- [`docs/generic_evb_workflow.md`](docs/generic_evb_workflow.md)
- [`docs/reference_profile_schema.md`](docs/reference_profile_schema.md)
- [`docs/irc_window_generation.md`](docs/irc_window_generation.md)
- [`docs/q_region_representation.md`](docs/q_region_representation.md)
- [`docs/umbrella_pmf_analysis.md`](docs/umbrella_pmf_analysis.md)
- [`docs/native_table_metadynamics.md`](docs/native_table_metadynamics.md)
- [`docs/migration_from_hg317_prototype.md`](docs/migration_from_hg317_prototype.md)

## Examples

Generic templates live in `examples/generic_*`. The former enzyme-specific research prototype is now an example/tutorial dataset under `examples/hg317/`; package source code does not depend on that system.

## Install

```bash
pip install -e ".[openmm,test]"
```

For optional analysis and PLUMED extras:

```bash
pip install -e ".[openmm,analysis,plumed,test]"
```
