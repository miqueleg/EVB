# HG3.17 Q-Region Stability Validation

This validation checks the selected calibrated HG3.17 Q-region EVB candidate before any native OPES work.

Selected candidate:

- `local_pme_q_atoms_cutoff_0.8`
- `delta_alpha_kj_mol: 405.2455501867761`
- `h12_kj_mol: 431.6758380801819`
- calibrated target: g-xTB RC/TS/PROD profile

The selected model remains approximate relative to the legacy full-state PME endpoint-charge Hamiltonian. It uses a shared PME baseline plus local direct-space state corrections. The stability check verifies that this calibrated Q-region Hamiltonian is numerically usable for longer dynamics, not that it is exact relative to legacy full-state PME.

## Command

```bash
evb hg317-qregion-stability-check \
  --config outputs/hg317_qregion_gxtb_calibrated/selected_candidate/fitted_config.yaml \
  --output outputs/hg317_qregion_stability \
  --platform CUDA \
  --plain-steps 100000 \
  --table-metad-steps 100000
```

The `selected_candidate/fitted_config.yaml` path is resolved from `validation/selected_candidate.json` when the convenience file is not present.

For a quick CI/smoke run:

```bash
evb hg317-qregion-stability-check \
  --config outputs/hg317_qregion_gxtb_calibrated/selected_candidate/fitted_config.yaml \
  --output outputs/hg317_qregion_stability_quick \
  --platform CPU \
  --plain-steps 10000 \
  --table-metad-steps 10000 \
  --quick
```

The same workflow is available as:

```bash
python scripts/hg317_qregion_stability_check.py
```

## Outputs

Each run writes:

- `observables.csv`
- `final.pdb`
- `stability_summary.json`
- `stability_summary.md`

The parent output directory also writes:

- `stability_overall_summary.json`
- `stability_overall_summary.md`

Trajectory DCD output is not written by default; the stability checks are focused on compact observables and final structures.

## Observables

The command records:

- step and time
- `E_common`, `e1_Q`, `e2_Q`, `E1`, `E2`
- shifted gap
- unbiased and biased EVB lower-surface energy
- EVB weights `w1` and `w2`
- table bias
- temperature
- max force norm at report stride
- substrate COM displacement when substrate atoms are configured
- configured key distances

## Stability Meaning

A run is marked stable when:

- it completes the requested steps
- no NaNs are detected
- no force explosion is detected
- no catastrophic energy jump is detected
- substrate COM drift is below threshold, or no substrate atoms are configured
- final PDB writing succeeds

A run is marked sampling-promising when it is stable and either:

- `w2` enters the 0.2 to 0.8 mixing range, or
- the shifted gap approaches zero within the configured tolerance

## Before OPES

Native OPES should only be implemented after these longer checks are stable. OPES should reuse the native gap-bias infrastructure and the calibrated Q-region representation; it should not reintroduce OpenMM `app.Metadynamics` or `BiasVariable` for the gap.
