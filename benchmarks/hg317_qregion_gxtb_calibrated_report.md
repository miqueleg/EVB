# HG3.17 Q-Region g-xTB Calibrated Report

## Summary

This branch calibrates computable HG3.17 Q-region EVB candidates directly to the user-provided g-xTB RC/TS/PROD reaction profile. The previous full-state endpoint-charge shifted-gap mismatch is retained as a diagnostic concept, but it is not used as the veto criterion for these changed or approximate Q-region Hamiltonians.

No OPES implementation was added.

## Git SHA

Calibration and smoke were run before the final branch commit from `c97eddc6efcecca1ba1e44163e9a20212104967c`.

## Tests Run

- `.venv/bin/python -m pytest tests/test_hg317_gxtb_calibration.py -q`: 7 passed.
- `.venv/bin/python -m compileall -q src/kemp_evb tests scripts/hg317_qregion_gxtb_workflow.py`.

## Reference Profile

| state | Energy Eh | Abs kcal/mol | Rel to Unbound kcal/mol |
| --- | ---: | ---: | ---: |
| Unbound | -5550.032438 | -3482698.080 | 0.000000 |
| Bound | -5550.088621 | -3482733.336 | -35.255147 |
| RC | -5550.088209 | -3482733.077 | -34.996349 |
| TS | -5550.059301 | -3482714.937 | -16.856535 |
| PROD | -5550.147733 | -3482770.429 | -72.348270 |

## Calibration Target

Default profile: `relative_to_RC`.

| state | target kJ/mol |
| --- | ---: |
| RC | 0.000000 |
| TS | 75.896981 |
| PROD | -156.280437 |

Full-system relaxed IRC frames were found from `prep/hg317_full_irc/evb_ready/setup_from_irc/analysis/irc_relaxed_seeds` using canonical frames 0, 121, and 242 for RC, TS, and PROD.

## Candidates Calibrated

| candidate | mode | delta_alpha_old | delta_alpha_new | H12_old | H12_new | RMS residual | max residual | exploratory_valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| local_pme_all_atoms_cutoff_1.2 | local_pme_approx | 307.198300 | 375.539409 | 568.434202 | 471.589060 | 1.03e-09 | 1.43e-09 | true |
| local_pme_q_atoms_cutoff_0.8 | local_pme_approx | 307.198300 | 405.245550 | 568.434202 | 431.675838 | 6.35e-10 | 1.07e-09 | true |
| local_pme_q_atoms_cutoff_1.0 | local_pme_approx | 307.198300 | 395.102362 | 568.434202 | 421.123616 | 8.40e-10 | 1.43e-09 | true |
| local_pme_q_atoms_cutoff_1.2 | local_pme_approx | 307.198300 | 395.081240 | 568.434202 | 421.132872 | 9.41e-10 | 1.36e-09 | true |
| local_pme_q_atoms_cutoff_1.5 | local_pme_approx | 307.198300 | 395.053971 | 568.434202 | 421.106099 | 9.12e-10 | 1.37e-09 | true |
| local_pme_q_plus_shell_cutoff_1.2 | local_pme_approx | 307.198300 | 402.569503 | 568.434202 | 447.733473 | 1.40e-09 | 2.42e-09 | true |
| local_pme_q_plus_shell_cutoff_1.5 | local_pme_approx | 307.198300 | 54.226845 | 568.434202 | 0.000000 | 20.7130 | 25.3681 | false |
| local_pme_q_plus_shell_cutoff_2.0 | local_pme_approx | 307.198300 | 445.259608 | 568.434202 | 578.395077 | 1.64e-09 | 2.70e-09 | true |
| shared_nonbonded_state1 | shared_nonbonded_model | 307.198300 | -131.404294 | 568.434202 | 394.103966 | 1.02e-09 | 1.65e-09 | true |
| shared_nonbonded_state2 | shared_nonbonded_model | 307.198300 | 6927.253525 | 568.434202 | 0.000000 | 14646.2855 | 20727.1094 | false |

## Candidate Selection

Selected candidate: `local_pme_q_atoms_cutoff_0.8`.

Selection reason: lowest fitted g-xTB profile RMS residual among force-sane, non-duplicated candidates. It is `exploratory_valid: true` and `production_valid: true` under the implemented profile/force thresholds, but it remains an approximate local-PME Hamiltonian and should not be described as exact relative to the legacy full-state PME endpoint-charge model.

Fitted parameters:

- `delta_alpha_kj_mol`: 405.2455501867761
- `h12_kj_mol`: 431.6758380801819
- RMS residual: 6.352383406599819e-10 kJ/mol
- Max residual: 1.070503685696167e-09 kJ/mol

## Smoke Run

| candidate | steps/s | ns/day | stable | duplicated_full_nonbonded | approximation | model_changed |
| --- | ---: | ---: | --- | --- | --- | --- |
| local_pme_q_atoms_cutoff_0.8 | 2235.368 | 48.284 | true | false | true | false |

Smoke details:

- steps: 2000
- platform: CUDA
- context creation time: 4.036836572922766 s
- MD wall time: 0.8947071270085871 s
- table bias updates: 2
- average table update time: 0.00016460660845041275 s
- `parameters_refit`: true
- forced despite failed validation: false

## Status

At least one HG3.17 Q-region native table-metad smoke trajectory ran.

For Q-region candidates in this workflow:

- `duplicated_full_nonbonded`: false
- `app.Metadynamics`/`BiasVariable`: not used by the native table-metad path
- selected candidate uses approximation: true
- selected candidate changes the nonbonded model: false
- selected candidate ignores/approximates reciprocal PME differences: true

## Scientific Caution

The selected local-PME candidate is calibrated to the g-xTB RC/TS/PROD profile and is numerically stable in a short smoke trajectory. It is still approximate relative to the legacy full-state PME endpoint-charge Hamiltonian. This branch makes HG3.17 Q-region sampling computable and calibrated, but longer stability checks and sampling validation are still required before production conclusions.

## Next Branch

Recommended next branch: `feature/native-opes-gap`, after confirming longer native table-metad stability for the selected calibrated candidate.
