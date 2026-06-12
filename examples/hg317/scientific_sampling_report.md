# HG3.17 Q-region scientific sampling correction report

## Summary

The previous Q-region reproduction run is a fail run. The quick umbrella and MetaD barriers were both 0 kcal/mol against the g-xTB RC-to-TS target of 18.139814 kcal/mol. The stability runs also stayed on the reactant-side shifted gap range and did not cross into the product basin.

This update replaces the nonphysical quick umbrella proxy with a true Q-region umbrella bias inside the same EVB lower-surface `CustomCVForce`, uses the relaxed IRC RC/TS/PROD full-system seed structures, audits multiple Q-region nonbonded candidates, and prepares TS-focused umbrella and native table MetaD inputs.

No OPES implementation was added.

## What went wrong

- The old quick umbrella analysis was a histogram/proxy, not a real biased umbrella PMF.
- The MD started from state-like coordinates whose initial shifted gap was not the calibrated RC seed gap.
- The 100k stability runs never crossed the fitted reaction gap path: the shifted gap remained negative, so no product-side basin was sampled.
- The `sampling_promising` flag was too optimistic because `w2` entered mixed values mainly due to large `H12`, not because the trajectory sampled the TS/product side.

## Corrective implementation

- `QRegionSystemBuilder.build()` now accepts `gap_umbrella_center` and `gap_umbrella_force_constant`.
- The umbrella term is added directly to the Q-region EVB `CustomCVForce` as `0.5*k_gap*(shifted_gap-gap_center)^2`.
- New command: `evb hg317-qregion-scientific-sampling`.
- New module: `src/kemp_evb/hg317_scientific_sampling.py`.
- New documentation: `docs/hg317_qregion_scientific_sampling.md`.

## Candidate gap audit

Short setup run: `outputs/hg317_qregion_scientific_sampling_quick3`.

| candidate | RC gap kcal/mol | TS gap kcal/mol | PROD gap kcal/mol |
| --- | ---: | ---: | ---: |
| local_pme_q_atoms_cutoff_0.8 | -763.787 | 105.352 | 1685.988 |
| local_pme_q_atoms_cutoff_1.2 | -764.626 | 107.502 | 1685.629 |
| local_pme_all_atoms_cutoff_1.2 | -773.096 | 96.607 | 1687.195 |
| local_pme_q_plus_shell_cutoff_2.0 | -808.553 | 70.329 | 1690.347 |
| shared_nonbonded_state1 | -736.801 | 113.445 | 1685.193 |

The selected `local_pme_q_atoms_cutoff_0.8` candidate is not obviously worse than broader local PME candidates in short dynamics and remains faster/cleaner. The `all_atoms` local correction shifts the TS gap by roughly 9 kcal/mol but does not clearly improve short screen behavior enough to replace the selected candidate yet.

## Short umbrella screen

The scientifically relevant test is whether a real restrained Q-region window tracks its center and can cover the gap=0/TS region from IRC seeds.

For `local_pme_q_atoms_cutoff_0.8`, 250-step diagnostic windows showed:

| seed | center kJ/mol | k_gap | gap range kJ/mol | positive frames | near-zero frames | interpretation |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| TS | 0 | 0.0015 | -326.7 to 440.8 | 5/11 | 7/11 | good zero-gap overlap screen |
| TS | 0 | 0.0030 | -178.3 to 440.8 | 7/11 | 10/11 | stronger, promising but monitor bias stiffness |
| TS | 440.8 | 0.0015 | 372.8 to 512.5 | 11/11 | 0/11 | good TS-centered tracking |
| RC | RC gap | 0.0015 | reactant-side only | 0/11 | 0/11 | expected reactant basin seed |

Recommended pilot umbrella settings:

- use the generated TS-focused window CSV;
- start with `k_gap = 0.0015 kJ/mol/(kJ/mol)^2`;
- include denser windows around gap=0 and the fitted TS gap;
- use IRC seed propagation rather than generic state1 coordinates;
- inspect adjacent overlap before treating the PMF as valid.

## Short native table MetaD screen

For the selected candidate:

| seed | width kJ/mol | height kJ/mol | gap range kJ/mol | interpretation |
| --- | ---: | ---: | --- | --- |
| RC | 1000 | 0.1 | -3195.7 to -133.6 | reaches near zero but not product side in 250 steps |
| TS | 1000 | 0.1 | -632.3 to 936.9 | best conservative TS crossing screen |
| TS | 750 | 0.2 | 440.8 to 4116.7 | too aggressive product-side push |
| TS | 500 | 0.5 | 440.8 to 3128.1 | too aggressive product-side push |

Recommended pilot MetaD settings:

- run RC-seeded and TS-seeded replicas;
- `bias_width = 1000 kJ/mol`;
- `height = 0.1 kJ/mol`;
- `bias_factor = 15`;
- keep table range `[-25000, 25000] kJ/mol` until pilot confirms safe narrowing;
- reject runs that do not visit both gap signs or that leave the table grid.

## Generated inputs

From `outputs/hg317_qregion_scientific_sampling_quick3`:

- `configs/qregion_scientific_gap_umbrella.yaml`
- `configs/qregion_scientific_gap_umbrella_windows.csv`
- `configs/qregion_scientific_gap_table_metad.yaml`
- `short_screen/umbrella_screen.csv`
- `short_screen/metad_screen.csv`
- `short_screen/short_screen_summary.json`
- `run_commands/run_short_screen.sh`
- `run_commands/prepare_production_inputs.sh`

## Production preparation command

```bash
.venv/bin/evb hg317-qregion-scientific-sampling \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_scientific_sampling_production \
  --platform CUDA \
  --mode production \
  --no-run-short-screen
```

## Scientific status

The earlier Q-region PMF/MetaD run should be rejected. The corrected setup now has the right ingredients for a scientifically meaningful exploration: relaxed IRC seeds, true Q-region gap umbrellas, native table MetaD, short screens for force constants and Gaussian shapes, and explicit failure criteria. Production validation still requires real overlap/convergence analysis and a barrier comparison against g-xTB and any available numeric legacy PMF outputs.
