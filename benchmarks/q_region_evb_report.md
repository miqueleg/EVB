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

- `python -m compileall -q src/kemp_evb tests`: passed
- `python -m pytest -q`: 88 passed, 2 skipped
- `python -m pytest tests/test_q_region_evb.py -q`: 18 passed

## Q6-Style Design Summary

The implementation builds:

- `E_common`: outer-system baseline/common forces evaluated once.
- `e1_Q`, `e2_Q`: state-specific residual forces inside the EVB `CustomCVForce`.
- Optional native `gap_bias(e1_Q - e2_Q - delta_alpha)` in the same EVB force.

Unsupported custom bonded incompatibilities, changed terms outside the Q/correction region, and differing Q-involved constraints fail rather than silently falling back to duplicated full states.

## Bonded Term Mapping Update

This branch replaces the previous bonded term-count equality restriction with canonical Q-region term mapping. Supported classes are `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`, `RBTorsionForce`, `CustomBondForce`, `CustomAngleForce`, and `CustomTorsionForce` when custom expressions and parameter definitions are compatible.

Mapping behavior:

- exact full-signature matches go to `E_common`;
- state1-only terms go to `e1_Q`;
- state2-only terms go to `e2_Q`;
- same atom key with changed parameters goes to both state residuals;
- repeated torsions on the same tuple are matched as a multiset;
- reversed bond/angle/torsion order is canonicalized where physically equivalent.

New tests cover state1-only and state2-only bonds, appearing/disappearing angles and torsions, changed bond parameters, repeated torsions, reversed angle/torsion order, outside-Q failures, and HG3.17 derivation smoke diagnostics.

HG3.17 derivation now reports detailed bonded mappings instead of the old generic failure:

- changed bonded terms: 32
- state1-only: 18
- state2-only: 4
- changed-parameter: 10
- by kind: 6 harmonic bonds, 10 harmonic angles, 16 periodic torsions
- old `Q-region changed HarmonicBondForce term count differs` message present: no

Constraint handling was also audited. Identical constraints are retained. Differing constraints involving Q or correction atoms fail by default with `q_atom_constraint_policy: fail`; this branch does not silently remove constraints.

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

CPU, 5000 steps, 3 repeats, bonded-mapping diagnostics enabled:

| mode | mean steps/s | mean ns/day |
| --- | ---: | ---: |
| legacy_full_state | 2704.056 | 233.630 |
| q_region_exact_direct | 2766.912 | 239.061 |
| q_region_table_metad | 2634.356 | 227.608 |

CUDA, 10000 steps, 3 repeats, bonded-mapping diagnostics enabled:

| mode | mean steps/s | mean ns/day |
| --- | ---: | ---: |
| legacy_full_state | 21480.010 | 1855.873 |
| q_region_exact_direct | 21453.320 | 1853.567 |
| q_region_table_metad | 18565.183 | 1604.032 |

Toy validation errors from benchmark records:

- max energy error: 0.0 kJ/mol
- max gap error: 0.0 kJ/mol
- max force RMSD: 0.0 kJ/mol/nm

Benchmark files:

- `benchmarks/q_region_bonded_mapping_toy_cpu.json`
- `benchmarks/q_region_bonded_mapping_toy_cuda.json`

## HG3.17 Status

Prepared files were present. `evb derive-q-region` wrote:

- `outputs/hg317_q_region_bonded_mapping/q_region_derivation_report.json`
- `outputs/hg317_q_region_bonded_mapping/q_region_bonded_mapping_report.json`
- `outputs/hg317_q_region_bonded_mapping/q_region_nonbonded_report.json`
- `outputs/hg317_q_region_bonded_mapping/hg317_q_region_config.yaml`

The old generic HarmonicBondForce term-count failure is resolved. `evb q-region-singlepoint` now writes bonded/nonbonded diagnostics and stops at the scientifically specific exact PME limitation:

```text
Q-region exact PME decomposition is not implemented; use full-state exact mode or explicitly enable local_pme_approx with validation.
```

No HG3.17 Q-region production benchmark was run because exact validation did not pass. This branch therefore does not claim HG3.17 Q-region production readiness.

## Duplicated Nonbonded Status

For supported toy Q-region modes, `duplicated_full_nonbonded` is false. For HG3.17, Q-region validation has not passed, so no production claim is made. Legacy/exact-decomposition modes may still report duplicated full PME.

## Bias API Status

Q-region table metadynamics avoids OpenMM `app.Metadynamics` and `BiasVariable`.

## Scientific Limitations

- Exact Q-region PME decomposition is not implemented.
- Local PME correction is approximate and disabled by default.
- Production use requires legacy validation on representative frames.
- Exact Q-region validation for HG3.17 is currently blocked by differing PME, not by bonded term-count mapping.

## Next Recommended Branch

`feature/native-opes-gap`, using native table bias plus the Q-region representation after system-specific validation, or a separate exact/validated nonbonded baseline branch for HG3.17 PME handling.
