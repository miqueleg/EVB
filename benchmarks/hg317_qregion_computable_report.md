# HG3.17 Q-Region Computable Candidate Report
## Summary
This branch adds computable HG3.17 Q-region candidate Hamiltonians for the PME-blocked case: `shared_nonbonded_model` and `local_pme_approx`. The legacy full-state EVB remains the validation reference. No OPES is implemented.
## Git SHA
Benchmarks and validation were run before the final feature commit, from `aff8a004577f3b7fcdd44118aafdba90fd0723eb`.
## Tests Run
- `python -m compileall -q src/kemp_evb tests scripts/benchmark_hg317_qregion_candidates.py`
- `python -m pytest tests/test_q_region_evb.py tests/test_q_region_hg317_candidates.py -q`
- `python -m pytest -q`: 97 passed, 2 skipped.
## Candidate Modes Implemented
- `full_state_reference`: legacy full-state EVB, slow validation reference.
- `shared_nonbonded_model`: one shared PME baseline, exact for a modified Hamiltonian, `legacy_equivalence: false`.
- `local_pme_approx`: one shared PME baseline plus local direct-space residuals and changed exception residuals, approximate relative to legacy PME.
## Candidate Configs Generated
Generated under `outputs/hg317_qregion_candidates/candidates/`:
- `local_pme_all_atoms_cutoff_1.2.yaml`
- `local_pme_q_atoms_cutoff_0.8.yaml`
- `local_pme_q_atoms_cutoff_1.0.yaml`
- `local_pme_q_atoms_cutoff_1.2.yaml`
- `local_pme_q_atoms_cutoff_1.5.yaml`
- `local_pme_q_plus_shell_cutoff_1.2.yaml`
- `local_pme_q_plus_shell_cutoff_1.5.yaml`
- `local_pme_q_plus_shell_cutoff_2.0.yaml`
- `shared_nonbonded_state1.yaml`
- `shared_nonbonded_state2.yaml`
## Q-Region IRC Fit Results
- `shared_nonbonded_state1`: fit_success=False; Calibration data with min1/min2/ts coordinates is required for a full Q-region IRC refit; original parameters were retained.
- `shared_nonbonded_state2`: fit_success=False; Calibration data with min1/min2/ts coordinates is required for a full Q-region IRC refit; original parameters were retained.
## Candidate Validation Table
| candidate | exactness_status | PME approx | model changed | max energy error | max gap error | force RMSD | force max abs | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| local_pme_all_atoms_cutoff_1.2 | approximate | True | False | 40.2445 | 33298 | 3.29569 | 398.964 | False |
| local_pme_q_atoms_cutoff_0.8 | approximate | True | False | 42.2626 | 33568.1 | 3.42659 | 433.699 | False |
| local_pme_q_atoms_cutoff_1.0 | approximate | True | False | 42.1718 | 33556.4 | 3.42034 | 432.107 | False |
| local_pme_q_atoms_cutoff_1.2 | approximate | True | False | 42.1723 | 33556.4 | 3.42034 | 432.107 | False |
| local_pme_q_atoms_cutoff_1.5 | approximate | True | False | 42.1718 | 33556.4 | 3.42034 | 432.107 | False |
| local_pme_q_plus_shell_cutoff_1.2 | approximate | True | False | 40.244 | 33298 | 3.29569 | 398.964 | False |
| local_pme_q_plus_shell_cutoff_1.5 | approximate | True | False | 39.3873 | 33176.6 | 3.24227 | 398.002 | False |
| local_pme_q_plus_shell_cutoff_2.0 | approximate | True | False | 41.8796 | 33518.5 | 3.40094 | 426.926 | False |
| shared_nonbonded_state1 | exact_for_shared_nonbonded_model | False | True | 39.7336 | 33226.1 | 3.38831 | 443.775 | False |
| shared_nonbonded_state2 | exact_for_shared_nonbonded_model | False | True | 33186.4 | 33226.1 | 12775 | 2.16428e+06 | False |
## Selected Best Candidate
`local_pme_q_plus_shell_cutoff_1.5` was selected as the best failed candidate because no candidate passed. It is `approximate`, has max gap error `33176.553374277486` kJ/mol, and force RMSD `3.2422726051417317` kJ/mol/nm.
## Smoke Benchmark
- ran: false
- reason: No candidate passed validation thresholds; use force_smoke to override.
## Nonbonded/Duplicated PME Status
All buildable Q-region candidates reported `duplicated_full_nonbonded: false`. Shared-nonbonded candidates changed the Hamiltonian. Local-PME candidates used an explicit approximation.
## Scientific Caution
HG3.17 is computable in Q-region mode in the sense that candidate systems now build and validate against the full-state reference. However, no candidate passed the current validation/refit criteria. The shared-nonbonded model needs a real IRC/reference refit before use; local-PME candidates showed very large gap errors versus the legacy endpoint-charge model. No production claim is made.
## Next Branch
Recommended next branch: `feature/native-opes-gap` only after selecting/refitting an acceptable Q-region candidate, or a dedicated nonbonded-model calibration branch for HG3.17.
