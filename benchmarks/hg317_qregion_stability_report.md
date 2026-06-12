# HG3.17 Q-Region Stability Report

## Summary

This branch validates the selected calibrated HG3.17 Q-region EVB candidate before native OPES development. It adds a stability command and runs both plain Q-region EVB MD and native table-metad Q-region MD for 100,000 steps on CUDA.

No OPES implementation was added.

## Git SHA

Stability checks were run before the final feature commit from `5f607057113d4a1690bc2c1d0dbf041ace84b4d4`.

## Tests Run

- `.venv/bin/python -m pytest tests/test_hg317_stability.py -q`: 3 passed.
- `.venv/bin/python -m compileall -q src/kemp_evb tests scripts/hg317_qregion_stability_check.py`: passed.
- `.venv/bin/python -m pytest -q`: 107 passed, 2 skipped.
- Direct CLI quick run on toy Q-region config: passed.

## Selected Candidate

- candidate: `local_pme_q_atoms_cutoff_0.8`
- `delta_alpha_kj_mol`: 405.2455501867761
- `h12_kj_mol`: 431.6758380801819
- calibration target:
  - RC = 0.000000 kJ/mol
  - TS = 75.896981 kJ/mol
  - PROD = -156.280437 kJ/mol
- exactness status: approximate
- PME treatment: shared PME baseline plus local direct-space state corrections
- `duplicated_full_nonbonded`: false

## Plain MD Stability

| metric | value |
| --- | ---: |
| steps | 100000 |
| stable | true |
| sampling promising | true |
| NaN detected | false |
| force explosion detected | false |
| catastrophic energy jump detected | false |
| substrate COM drift max | 0.125320 nm |
| max force norm | 7884.044 kJ/mol/nm |
| max energy jump | 6577.786 kJ/mol |

## Native Table-Metad Stability

| metric | value |
| --- | ---: |
| steps | 100000 |
| stable | true |
| sampling promising | true |
| NaN detected | false |
| force explosion detected | false |
| catastrophic energy jump detected | false |
| substrate COM drift max | 0.134362 nm |
| max force norm | 7884.044 kJ/mol/nm |
| max energy jump | 6409.201 kJ/mol |
| bias updates | 100 |
| average bias update time | 0.000102935 s |

## Speed

| run | steps/s | ns/day |
| --- | ---: | ---: |
| plain_md | 2216.969 | 47.887 |
| table_metad | 2221.431 | 47.983 |

## Gap And Mixing Summary

| run | shifted gap min | shifted gap max | shifted gap mean | shifted gap std | w2 min | w2 max | mixing frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain_md | -6464.859 | -92.330 | -470.199 | 610.436 | 0.004400 | 0.446831 | 98 |
| table_metad | -6464.859 | -163.019 | -469.048 | 607.059 | 0.004400 | 0.407229 | 100 |

Both runs satisfied the mixing-region criterion because `w2` entered the 0.2 to 0.8 range. The shifted gap approached zero within the configured 250 kJ/mol tolerance in both runs.

## OPES Readiness

The calibrated Q-region candidate is ready for native OPES development from a numerical-stability standpoint:

- both 100k-step checks completed
- no NaNs were detected
- final structures were written
- `duplicated_full_nonbonded` remained false
- the mixing region was visited
- native table updates were inexpensive

Scientific caution remains: the selected candidate is a `local_pme_approx` Hamiltonian. It is calibrated to the g-xTB profile and stable in these checks, but it is not exact relative to the legacy full-state PME endpoint-charge model.

## Next Branch

Recommended next branch: `feature/native-opes-gap`.
