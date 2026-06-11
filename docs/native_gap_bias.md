# Native EVB Gap Table Bias

`evb-gap-metad` keeps the original OpenMM `app.Metadynamics` path. That path creates a separate `BiasVariable` for the EVB gap, so OpenMM evaluates a gap CV force in addition to the EVB lower-surface `CustomCVForce`. The two evaluations are algebraically consistent, but they are separate `CustomCVForce` inner-context evaluations.

`evb-gap-table-metad` adds the bias directly to the EVB lower-surface force:

```text
E_total = E_common + lower_EVB(e1, e2, delta_alpha, h12) + gap_bias(e1 - e2 - delta_alpha)
```

For exact decomposed systems, `common_force_placement: outer_system` adds common forces directly to the outer OpenMM `System` and keeps only the state-specific residual energies inside the EVB `CustomCVForce`:

```text
lower_EVB = 0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + h12^2)
```

The bias term is a mutable `Continuous1DFunction` named `gap_bias`. Python updates the grid periodically, then calls `updateParametersInContext()` on the EVB force. This is the intended infrastructure for native OPES/OPES-Vk because future adaptive methods can update the same table without building a second gap CV force.

## Commands

- `evb-gap-metad`: existing OpenMM `app.Metadynamics`/`BiasVariable` implementation.
- `evb-gap-table-metad`: new native table-bias implementation. It does not call `openmm.app.Metadynamics` and does not create a `BiasVariable`.

Example YAML:

```yaml
sampling:
  mode: gap_table_metadynamics
  native_gap_bias:
    method: well_tempered_metadynamics
    cv: gap
    min_value: -25000.0
    max_value: 25000.0
    grid_width: 1000
    bias_width: 750.0
    height_kj_mol: 1.0
    bias_factor: 15.0
    frequency: 1000
    save_frequency: 10000
    bias_dir: native_bias
    restart: true
    wall_force_constant_kj_mol2: 0.000001
    update_scheme: table_in_context

evb:
  energy_decomposition:
    enabled: true
    mode: exact
    common_force_placement: outer_system
```

If `native_gap_bias` values are omitted, the new command reuses the existing `sampling.metadynamics` values where possible. This fallback is only for `evb-gap-table-metad`; it does not change `evb-gap-metad`.

## Restart Files

The native path writes these files under `bias_dir`:

- `native_gap_bias_state.json`: restartable grid metadata and values.
- `native_gap_bias_table.csv`: current bias table for inspection.

The observable log is `gap_table_metad_colvar.csv` and includes the shifted gap, table bias, residual diabatic energies, common energy when available, full E1/E2, unbiased and biased EVB energies, and EVB weights.

## Exactness Limits

This branch does not introduce approximate local nonbonded corrections. It also does not solve duplicated full PME when the two diabatic states contain different full-system `NonbondedForce` objects. If diagnostics report `duplicated_full_nonbonded: true`, both full PME states are still evaluated.

## Benchmarks

Toy CPU benchmark:

```bash
python scripts/benchmark_native_gap_bias.py \
  --config examples/toy_evb.yaml \
  --platform CPU \
  --steps 5000 \
  --repeats 3 \
  --modes plain,app-metad,table-metad \
  --output benchmarks/native_gap_bias_toy_cpu.json
```

CUDA benchmark, when available:

```bash
python scripts/benchmark_native_gap_bias.py \
  --config examples/toy_evb.yaml \
  --platform CUDA \
  --steps 10000 \
  --repeats 3 \
  --modes plain,app-metad,table-metad \
  --output benchmarks/native_gap_bias_toy_cuda.json
```

HG3.17 benchmark, only when inputs are present:

```bash
python scripts/benchmark_native_gap_bias.py \
  --config examples/hg317_evb_gap_metad.yaml \
  --platform CUDA \
  --steps 2000 \
  --repeats 3 \
  --modes app-metad,table-metad \
  --output benchmarks/native_gap_bias_hg317_cuda.json
```
