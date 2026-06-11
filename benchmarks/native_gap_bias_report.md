# Native Gap Bias Report

## Summary

This branch adds native tabulated EVB gap bias infrastructure through a mutable `gap_bias(gap)` function inside the same EVB lower-surface `CustomCVForce`. The new command is `evb-gap-table-metad`; the existing `evb-gap-metad` command remains the OpenMM `app.Metadynamics`/`BiasVariable` path.

The new table-metad path removes the separate OpenMM `app.Metadynamics`/`BiasVariable` gap-bias evaluation. It does not remove duplicated full PME when two exact decomposed diabatic states still contain differing full-system PME `NonbondedForce` objects.

## Git SHA

Benchmark runs were taken before the final feature commit:

- `4f6fc2efb155eb60aaded0b66c18e5cde549b777`

## Environment

- Python: 3.12.3
- OpenMM: 8.5.2.dev-36a30cb
- Platforms available: Reference, CPU, CUDA, OpenCL

## Tests

- `python -m compileall -q src scripts`: passed
- `python -m pytest -q`: 70 passed, 2 skipped
- `python -m kemp_evb.cli evb-gap-table-metad --config /tmp/toy_gap_table_cli.yaml`: passed

## Toy Accuracy Results

Covered by `tests/test_native_gap_bias.py`:

- Zero-bias native table EVB matches unbiased EVB energy, shifted gap, and forces with `1e-6` tolerances.
- Constant +3.0 kJ/mol table shifts energy by +3.0 kJ/mol and leaves forces unchanged.
- Harmonic table bias agrees with native gap umbrella energy, force, and gap checks.
- Gaussian deposition changes the bias table; restart JSON reload reproduces the same bias energy.
- Decomposed `common_force_placement: outer_system` matches `cv_compatible` energies and forces, and E1/E2 reporting remains correct.
- The new table path does not use `openmm.app.Metadynamics` or `BiasVariable`.

## Benchmarks

Toy CPU, `examples/toy_evb.yaml`, 5000 steps, 3 repeats:

| mode | mean steps/s | mean ns/day | app.Metadynamics | BiasVariable |
| --- | ---: | ---: | --- | --- |
| plain | 2531.140 | 218.691 | false | false |
| app-metad | 1556.763 | 134.504 | true | true |
| table-metad | 2597.117 | 224.391 | false | false |

Toy CUDA, `examples/toy_evb.yaml`, 10000 steps, 3 repeats:

| mode | mean steps/s | mean ns/day | app.Metadynamics | BiasVariable |
| --- | ---: | ---: | --- | --- |
| plain | 21495.809 | 1857.238 | false | false |
| app-metad | 9625.365 | 831.632 | true | true |
| table-metad | 21418.115 | 1850.525 | false | false |

HG3.17 CUDA smoke benchmark, `examples/hg317_evb_gap_metad.yaml`, 2000 steps, 3 repeats:

| mode | mean steps/s | mean ns/day | app.Metadynamics | BiasVariable |
| --- | ---: | ---: | --- | --- |
| app-metad | 921.339 | 19.901 | true | true |
| table-metad | 1887.900 | 40.779 | false | false |

Benchmark files:

- `benchmarks/native_gap_bias_toy_cpu.json`
- `benchmarks/native_gap_bias_toy_cuda.json`
- `benchmarks/native_gap_bias_hg317_cuda.json`

## HG3.17 Status

HG3.17 prepared files were present, so the short CUDA smoke benchmark ran. The benchmark above used the configured native gap metadynamics example and is not a production sampling claim.

This branch does not solve the exact-decomposition duplicated PME issue. If exact decomposed HG3.17 systems report `duplicated_full_nonbonded: true`, both full state-specific PME `NonbondedForce` objects are still evaluated. No approximate local nonbonded correction was introduced.

## Next Steps

Native OPES can now use this table-bias infrastructure. Local nonbonded correction or a shared nonbonded baseline remains a separate later branch.
