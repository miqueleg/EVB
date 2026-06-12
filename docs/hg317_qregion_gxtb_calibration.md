# HG3.17 Q-Region g-xTB Calibration

This workflow calibrates computable HG3.17 Q-region EVB candidates directly to the user-provided g-xTB reaction profile. The old full-state endpoint-charge EVB model remains useful as a diagnostic, but it is no longer the calibration target for Q-region candidate Hamiltonians.

## Reference Profile

The machine-readable reference is `examples/hg317_gxtb_reference_profile.yaml`.

The default target is the RC-relative profile:

| state | kcal/mol | kJ/mol |
| --- | ---: | ---: |
| RC | 0.000000 | 0.000000 |
| TS | 18.139814 | 75.896981 |
| PROD | -37.351921 | -156.280437 |

The file also includes Unbound and Bound absolute/relative values. Bound is reported as a diagnostic because Bound and RC differ by only about 0.259 kcal/mol.

## Why Raw Legacy Gap Mismatch Is Not A Veto

`shared_nonbonded_model` deliberately changes the Hamiltonian by using one shared PME baseline. `local_pme_approx` keeps a shared PME baseline and adds approximate local direct-space corrections. Both can have large raw shifted-gap offsets relative to the legacy full-state endpoint-charge model.

Those offsets are not automatic failures. `delta_alpha` is the EVB energy offset and can absorb large gap shifts. The calibrated model is judged against the g-xTB RC/TS/PROD lower-surface profile, plus numerical stability, force sanity, and diagnostics such as `duplicated_full_nonbonded`.

## Fitting

For each candidate, the workflow evaluates Q-region diabatic energies on full-system RC, TS, and PROD frames. It fits:

- `delta_alpha_kj_mol`
- `h12_kj_mol`

The objective is the RMS residual between the EVB lower-surface relative profile and the g-xTB target. The fitter performs a wide grid search, a refined grid search, and an optional SciPy local optimization when SciPy is available.

Outputs include:

- `fit_scan_coarse.csv`
- `fit_scan_refined.csv`
- `fitted_parameters.json`
- `fitted_config.yaml`
- `fit_report.json`
- `profile_fit.csv`

## Frame Discovery

Full-system coordinates are required. The workflow searches the existing setup-from-IRC outputs first, including:

- `prep/hg317_full_irc/evb_ready/setup_from_irc/analysis/evb_reference_fit_from_irc.json`
- `prep/hg317_full_irc/evb_ready/setup_from_irc/analysis/irc_relaxed_seeds/irc_seed_relaxation.csv`

The raw `examples/HD3.17_IRC.xyz` file is cluster-only and is not used as full-system coordinates unless existing embedding/relaxation outputs provide corresponding full-system frames.

## Commands

Calibrate existing candidates:

```bash
evb calibrate-hg317-qregion-gxtb \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --candidate-dir outputs/hg317_qregion_candidates/candidates \
  --output outputs/hg317_qregion_gxtb_calibrated \
  --profile relative_to_RC \
  --platform CUDA
```

Validate fitted candidates:

```bash
evb validate-hg317-qregion-gxtb \
  --calibrated-dir outputs/hg317_qregion_gxtb_calibrated \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_gxtb_calibrated/validation \
  --platform CUDA
```

Run an exploratory smoke if a fitted candidate is `exploratory_valid`:

```bash
evb smoke-hg317-qregion-gxtb \
  --calibrated-dir outputs/hg317_qregion_gxtb_calibrated \
  --output outputs/hg317_qregion_gxtb_calibrated/smoke \
  --platform CUDA \
  --steps 2000 \
  --smoke-policy exploratory
```

One-shot workflow:

```bash
evb hg317-qregion-gxtb-workflow \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_gxtb_workflow \
  --profile relative_to_RC \
  --platform CUDA \
  --smoke-steps 2000 \
  --smoke-policy exploratory
```

The same flow is available through `scripts/hg317_qregion_gxtb_workflow.py`.

## Validity Labels

`production_valid` requires strict profile residuals and force sanity. This branch does not claim production validity unless those thresholds are met.

`exploratory_valid` means the model builds, fits the g-xTB profile within loose thresholds, avoids duplicated full PME, has finite forces, and does not exceed the catastrophic force threshold.

`--smoke-policy force` can run a forced smoke for a non-exploratory candidate. The output marks it as forced and non-production.

## Scope

This is not OPES. It provides calibrated Q-region candidate Hamiltonians and native table-metad smoke infrastructure. The next OPES branch should start only after an exploratory smoke is stable.
