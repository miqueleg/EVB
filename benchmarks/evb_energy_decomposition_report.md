# EVB Energy Decomposition Benchmark Report

## Summary

This branch adds an optional exact native EVB energy-decomposition path and keeps the legacy full-state implementation as the default. Exact mode evaluates

```text
E1 = E_common + e1
E2 = E_common + e2
E_EVB = E_common + 0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + H12^2)
gap = e1 - e2 - delta_alpha
```

It also adds automatic EVB-ready adiabatic-system setup from IRC-prepared state files via `evb prepare-adiabatic-system`, plus a conservative HG3.17/5RGE system-preparation scaffold via `evb prepare-hg317-system`.

## Git Commits

- Baseline benchmark commit: `e24af61054438371d7a76f8eaf15f3fe0813dd8c`
- First refactor/report commit: `107b173ca311e1fce28e294cc7716b901cba226d`
- Current report generated from working tree after adding automatic adiabatic setup and HG3.17 GPU workflow benchmarks.

## Environment

- Python: `3.12.3`
- OpenMM: `8.5.2.dev-36a30cb`
- CUDA support: available through `openmm-cuda-12==8.5.2`
- OpenMM platforms: `['Reference', 'CPU', 'CUDA', 'OpenCL']`
- GPU: `NVIDIA GeForce RTX 5090`, driver `580.159.03`, CUDA runtime package `12.9`
- Benchmark CUDA platform properties: `{'Precision': 'mixed'}`

## Tests

- Before change: `50 passed, 3 skipped`
- After energy-decomposition implementation: `53 passed, 2 skipped`
- Current final run after automatic adiabatic setup and HG3.17 workflow benchmarks: `53 passed, 2 skipped`
- Current toy CLI checks completed: `evb validate`, `evb singlepoint`, `evb minimize`, `evb sample-series`, `evb analyze`

## Automatic Adiabatic-System Setup

Implemented:

```bash
evb prepare-adiabatic-system   --config examples/hg317_irc_mapping_barrier_calibrated.yaml   --output prep/hg317_full_irc/evb_ready   --write-window-config
```

The command runs `prepare-evb-inputs`, derives a missing alpha-carbon mapping when possible from `inputs/5RGE.pdb` and the IRC fixed atoms, then runs `setup-from-irc`. On this machine it completed using CUDA and generated EVB-ready OpenMM bundles plus IRC-relaxed adiabatic seeds.

HG3.17 setup result:

- `delta_alpha_kj_mol`: `396.7667143696792`
- `h12_kj_mol`: `705.1316403576698`
- Relaxed IRC frames: `243`
- Platform used for IRC relaxation: `CUDA`
- Window generation status: `blocked_pathological_irc_scan`
- Largest adjacent shifted-gap jump: `14917.995 kJ/mol`
- Largest absolute shifted gap: `25044.481 kJ/mol`

The generated HG3.17 files were sufficient for short CUDA umbrella/metadynamics benchmarks, but the pathological gap scan means the generated windows should not be treated as production-quality sampling input without further scientific review.

## Fresh-Clone HG3.17/5RGE Preparation Status

The repo contains `inputs/5RGE.pdb`, `examples/HD3.17_IRC.xyz`, HG3.17 EVB configs, and matched draft `prmtop`/`inpcrd` files. A complete fresh-clone pipeline from raw PDB through final AMBER topologies is not fully present. This branch adds `evb prepare-hg317-system --config examples/hg317_system_prep.yaml` as a conservative scaffold.

The scaffold:

- Requires AmberTools executables when `--execute` is used: `antechamber`, `parmchk2`, and `tleap`.
- Optionally uses `pdbfixer` for PDB cleaning when installed.
- Allows explicit protonation choices and active-site water retention rules in YAML.
- Allows user-supplied ligand `mol2`/`sdf` files when automatic ligand perception is ambiguous.
- Writes intermediate files and a preparation report.
- Fails loudly if AM1-BCC, GAFF2, or tleap execution is requested but unavailable.

## Toy Benchmarks

| Run | Platform | Context s | steps/s | ns/day | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline CPU legacy | CPU | 0.0016 | 2310.3 | 199.61 | Pre-refactor |
| Baseline CUDA legacy | CUDA | 0.0809 | 21344.9 | 1844.2 | Pre-refactor |
| Refactor CPU legacy | CPU | 0.0017 | 2767.8 | 239.14 | Current legacy path |
| Refactor CPU decomposed | CPU | 0.0022 | 2180.7 | 188.41 | Exact decomposed path |
| Refactor CUDA legacy | CUDA | 0.1024 | 4835.0 | 417.74 | Toy, not production-representative |
| Refactor CUDA decomposed | CUDA | 0.1474 | 7276.2 | 628.66 | Toy speed ratio 1.50x |

Toy accuracy tests comparing legacy and decomposed mode passed with zero measured energy, shifted-gap, force RMSD, and force max differences in the deterministic test case.

## HG3.17 CUDA Mixed-Precision Accuracy

Same-coordinate singlepoint checks were run on the generated HG3.17 EVB-ready system.

| Check | Energy abs diff kJ/mol | Shifted gap abs diff kJ/mol | Mean force-norm abs diff kJ/mol/nm |
| --- | ---: | ---: | ---: |
| Plain EVB lower surface | 0.000063 | 0.000192 | 0.00000084 |
| Gap umbrella total energy | 0.001647 | 0.000060 | 0.00000209 |

The remaining nonzero differences are CUDA mixed-precision/summation-level differences on a 32,777-atom system. Small deterministic CPU tests enforce tighter exactness for energy, gap, and forces.

## HG3.17 CUDA Workflow Benchmarks

All HG3.17 workflow benchmarks used `--platform CUDA`, `--cuda-precision mixed`, `--steps 2000`, `--repeats 3`, `--no-forces`, and the stabilized short benchmark settings in `benchmarks/hg317_gap_umbrella_benchmark.yaml` and `benchmarks/hg317_gap_metad_benchmark.yaml`.

| Workflow | Mode | Context s | Minimize s | MD wall s | steps/s | ns/day | Common terms | State-specific terms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gap umbrella | Legacy | 0.600 | 1.839 | 1.142 | 1751.9 | 7.568 | 0 | 0 |
| Gap umbrella | Decomposed | 0.549 | 1.839 | 1.329 | 1504.6 | 6.500 | 33552 | 44 |
| Gap metadynamics | Legacy | 1.945 | 2.565 | 2.417 | 827.4 | 3.574 | 0 | 0 |
| Gap metadynamics | Decomposed | 8.273 | 2.416 | 2.632 | 759.9 | 3.283 | 33552 | 44 |

Speed ratios, decomposed over legacy:

- HG3.17 gap umbrella: `0.859x`
- HG3.17 native gap metadynamics: `0.918x`

The branch is usable on the GPU, but HG3.17 does not speed up yet because exact mode refuses to approximate differing PME `NonbondedForce` objects. The decomposition report for HG3.17 consistently warns:

```text
NonbondedForce differs between states and was kept in full state-specific EVB evaluation; exact local decomposition for PME is not implemented.
```

## Scientific Caution

Exact mode preserves the EVB physics by construction and does not silently introduce approximate nonbonded physics. Unsupported differing terms stay in the full state-specific EVB path and are reported. Differing `NonbondedForce` objects, including PME systems, are not decomposed by local approximations in this branch.

For HG3.17 specifically, the current generated adiabatic IRC scan is flagged as pathological for automated window placement. The short umbrella/metadynamics runs are implementation benchmarks, not validated production sampling.

## Next Step

The next recommended feature is native OPES on the EVB gap using a tabulated OpenMM bias. That is intentionally not implemented in this branch.
