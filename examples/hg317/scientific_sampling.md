# HG3.17 Q-region scientific umbrella and MetaD setup

The first Q-region PMF/MetaD reproduction run failed scientifically: both quick barriers were reported as 0 kcal/mol, while the g-xTB RC-to-TS target is 18.139814 kcal/mol. The failure should not be interpreted as a converged Q-region result. It exposed two workflow problems:

- the quick umbrella path was a histogram proxy, not a true biased Q-region umbrella calculation;
- the dynamics started from generic state coordinates rather than the calibrated full-system IRC RC/TS/PROD seeds.

This workflow fixes those issues by using a true Q-region gap umbrella term inside the same EVB `CustomCVForce` and by building window plans from the calibrated Q-region gaps at the relaxed IRC RC, TS, and PROD frames. It follows the original full-state EVB sampling philosophy from `master`: seed from IRC/relaxed windows, apply real gap restraints, propagate windows, and analyze only when overlap/convergence exists.

## Selected candidate

The current calibrated Q-region candidate remains:

- `local_pme_q_atoms_cutoff_0.8`
- `delta_alpha_kj_mol = 405.2455501867761`
- `h12_kj_mol = 431.6758380801819`

This is approximate relative to exact full-state endpoint-charge PME because it uses `local_pme_approx`. It is calibrated to the g-xTB RC/TS/PROD energy profile, not claimed exact relative to the old full-state PME Hamiltonian.

## Command

Short diagnostic screen:

```bash
.venv/bin/evb hg317-qregion-scientific-sampling \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_scientific_sampling_quick \
  --platform CUDA \
  --mode quick \
  --screen-steps 2000
```

Prepare production inputs without running screens:

```bash
.venv/bin/evb hg317-qregion-scientific-sampling \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_scientific_sampling_production \
  --platform CUDA \
  --mode production \
  --no-run-short-screen
```

## Window construction

The umbrella coordinate is the fitted shifted Q-region gap:

```text
shifted_gap = e1_Q - e2_Q - delta_alpha
```

The umbrella bias is applied inside the Q-region EVB lower-surface force:

```text
E = E_common + EVB(e1_Q, e2_Q) + 0.5*k_gap*(shifted_gap-gap_center)^2
```

The quick plan uses a reduced TS-focused grid. Pilot and production use denser grids between the calibrated RC, gap=0, TS, and PROD gaps. The default production force constant starts at `0.0015 kJ/mol/(kJ/mol)^2` and should be adjusted after overlap inspection.

## Native table MetaD

MetaD uses the native tabulated gap bias, not OpenMM `app.Metadynamics` or `BiasVariable`. The short screen compares conservative Gaussian choices. Early HG3.17 screens favor a broad, low hill for TS-seeded exploration:

- `bias_width = 1000 kJ/mol`
- `height = 0.1 kJ/mol`
- `bias_factor = 15`
- broad table range `[-25000, 25000] kJ/mol` until pilot data justify narrowing

Stronger settings (`width=750,height=0.2` and `width=500,height=0.5`) pushed TS-seeded trajectories too rapidly toward product-side gaps in sub-ps diagnostic runs.

## Scientific acceptance

A valid PMF/metadynamics result must show:

- sampling in both reactant and product gap basins;
- overlap through the gap=0/TS region;
- no NaNs or force explosions;
- stable substrate position and final structures;
- a barrier within the configured threshold: warning above 2 kcal/mol, fail above 3 kcal/mol versus g-xTB or numeric legacy reference.

Static g-xTB fitting alone is not PMF validation. A short run that does not sample both basins is a failed or incomplete sampling result, even if the parameter fit is exact at RC/TS/PROD.
