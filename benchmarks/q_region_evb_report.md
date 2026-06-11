# Q-Region EVB Report

## Summary

This branch adds a Q6-style Q-region EVB representation. It evaluates a shared baseline/common system once and mixes only Q-region residual energies in the EVB `CustomCVForce`. The Q-region path integrates with the native table-bias infrastructure and does not use OpenMM `app.Metadynamics` or `BiasVariable`.

## Git SHA

Benchmarks were run before the final feature commit:

- `26140382d22c916873b3ee26a4a6622684ecdc7b`

## Environment

- Python: 3.12.3
- OpenMM: 8.5.2.dev-36a30cb
- Platforms available: Reference, CPU, CUDA, OpenCL

## Tests

- `python -m compileall -q src scripts`: passed
- `python -m pytest -q`: 78 passed, 2 skipped

## Q6-Style Design Summary

The implementation builds:

- `E_common`: outer-system baseline/common forces evaluated once.
- `e1_Q`, `e2_Q`: state-specific residual forces inside the EVB `CustomCVForce`.
- Optional native `gap_bias(e1_Q - e2_Q - delta_alpha)` in the same EVB force.

Unsupported changed Q-region topologies fail rather than silently falling back to duplicated full states.

## Exactness Matrix

| Mode | Full-state duplicated PME? | Exact? | Intended use |
| --- | --- | --- | --- |
| legacy_full_state | yes | yes | reference |
| exact_decomposition | maybe | yes | audit/reference |
| q_region_exact_direct | no | yes for direct-space | production where valid |
| q_region_local_pme_approx | no | no | experimental, validated only |
| q_region_table_metad | depends on q_region mode | depends | enhanced sampling |

## Toy Accuracy Results

From tests and benchmark validation:

- Energy error vs legacy: 0 kJ/mol
- Gap error vs legacy: 0 kJ/mol
- Force RMSD vs legacy: 0 kJ/mol/nm
- Q-region table-bias zero and constant-bias tests pass.
- Differing PME exact mode fails clearly unless `local_approx_enabled` is explicitly set.
- Local PME approximation is marked approximate and does not claim exactness.

## Toy Benchmarks

CPU, 5000 steps, 3 repeats:

| mode | mean steps/s | mean ns/day |
| --- | ---: | ---: |
| legacy_full_state | 2373.049 | 205.031 |
| exact_decomposition | 2028.055 | 175.224 |
| q_region_exact_direct | 2483.451 | 214.570 |
| q_region_table_metad | 2358.526 | 203.777 |

CUDA, 10000 steps, 3 repeats:

| mode | mean steps/s | mean ns/day |
| --- | ---: | ---: |
| legacy_full_state | 21519.106 | 1859.251 |
| exact_decomposition | 15302.703 | 1322.154 |
| q_region_exact_direct | 21482.282 | 1856.069 |
| q_region_table_metad | 18577.131 | 1605.064 |

Benchmark files:

- `benchmarks/q_region_toy_cpu.json`
- `benchmarks/q_region_toy_cuda.json`

## HG3.17 Status

Prepared files were present. `evb derive-q-region` wrote:

- `outputs/hg317_q_region_derivation/q_region_derivation_report.json`
- `outputs/hg317_q_region_derivation/hg317_q_region_config.yaml`

Exact Q-region singlepoint validation did not pass and no HG3.17 Q-region benchmark was run. The failure was a clear unsupported-topology error:

```text
Q-region changed HarmonicBondForce term count differs; this is not supported yet.
```

This branch therefore does not claim HG3.17 Q-region production readiness.

## Duplicated Nonbonded Status

For supported toy Q-region modes, `duplicated_full_nonbonded` is false. For HG3.17, Q-region validation has not passed, so no production claim is made. Legacy/exact-decomposition modes may still report duplicated full PME.

## Bias API Status

Q-region table metadynamics avoids OpenMM `app.Metadynamics` and `BiasVariable`.

## Scientific Limitations

- Exact Q-region PME decomposition is not implemented.
- Local PME correction is approximate and disabled by default.
- Production use requires legacy validation on representative frames.
- Changed bonded topologies with differing term counts need additional Q-term mapping support.

## Next Recommended Branch

`feature/native-opes-gap`, using native table bias plus the Q-region representation after system-specific validation.
