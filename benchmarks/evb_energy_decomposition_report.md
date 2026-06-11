# EVB Energy Decomposition Benchmark Report

## Summary

This branch adds an optional exact native EVB energy-decomposition path. The legacy full-state EVB implementation remains the default. Exact mode builds a common energy contribution once and evaluates only differing terms inside the two diabatic residual energies. Unsupported differing forces remain in the full state-specific path and are reported.

## Git Commits

- Baseline benchmark commit: `e24af61054438371d7a76f8eaf15f3fe0813dd8c`
- Refactor/report commit at generation time: `35b8a74`

## Environment

- Python: `3.12.3`
- OpenMM: `8.5.2.dev-36a30cb`
- OpenMM platforms: `['Reference', 'CPU', 'CUDA', 'OpenCL']`
- GPU: `NVIDIA GeForce RTX 5090, 580.159.03, 32607 MiB`
- CUDA package installed: `openmm-cuda-12==8.5.2` with CUDA 12.9 runtime wheels

## Tests

- Before change: `50 passed, 3 skipped`
- After change: `53 passed, 2 skipped`
- Post-refactor toy CLI checks: `validate`, `singlepoint`, `minimize`, `sample-series`, and `analyze` completed successfully.
- HG3.17 prep scaffold smoke test: `evb prepare-hg317-system --config examples/hg317_system_prep.yaml` completed without executing AmberTools and reported missing tools.

## Benchmark Table

| Run | Platform | Context creation s | steps/s | ns/day | mean EVB energy kJ/mol | mean shifted gap kJ/mol |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline CPU legacy | CPU | 0.00159892 | 2310.3 | 199.61 | -2.24545 | -1.15172 |
| Baseline CUDA legacy | CUDA | 0.0808621 | 21344.9 | 1844.2 | -3.35937 | -1.56928 |
| Refactor CPU legacy | CPU | 0.00165848 | 2767.83 | 239.14 | -3.42055 | -1.40823 |
| Refactor CPU decomposed | CPU | 0.00218284 | 2180.67 | 188.41 | -3.2362 | -1.56756 |
| Refactor CUDA legacy | CUDA | 0.102374 | 4835 | 417.744 | -2.82 | -1.46949 |
| Refactor CUDA decomposed | CUDA | 0.14735 | 7276.16 | 628.66 | -2.91876 | -1.42292 |

## Accuracy: Legacy vs Decomposed

| Platform | Energy abs diff kJ/mol | Shifted gap abs diff kJ/mol | Force RMSD kJ/mol/nm | Force max abs kJ/mol/nm |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0 | 0 | 0 | 0 |
| CUDA | 0 | 0 | 0 | 0 |

## Speed Comparison

On the two-atom toy system, GPU timings are dominated by context/CV overhead and are not representative of production protein systems. The measured CUDA toy steps/s ratio, decomposed over legacy, was `1.50489`. The exact decomposition report for the toy system found no common terms because the single harmonic bond differs between states.

## HG3.17 Status

The repo contains `inputs/5RGE.pdb`, `examples/HD3.17_IRC.xyz`, HG3.17 EVB configs, and matched draft prmtop/inpcrd files. It does not contain the full EVB-ready OpenMM bundles needed for the requested HG3.17 benchmark:

- `prep/hg317_full_irc/evb_ready/state1_evb_ready/system.xml`
- `prep/hg317_full_irc/evb_ready/state2_evb_ready/system.xml`
- `prep/hg317_full_irc/evb_ready/state1_evb_ready/coordinates.pdb`
- `prep/hg317_full_irc/evb_ready/state2_evb_ready/coordinates.pdb`

The existing checked-in HG3.17 preparation report states that the transferred-proton topology remains a blocking issue for a fresh-clone reproducible workflow. This branch therefore adds `evb prepare-hg317-system --config examples/hg317_system_prep.yaml` as a conservative scaffold. It writes cleaned PDBs, IRC-derived ligand seeds, AmberTools/tleap drivers, and a preparation report, and fails loudly if `--execute` is requested without `antechamber`, `parmchk2`, and `tleap`.

## Scientific Caution

Exact mode preserves the EVB physics by construction. It does not silently approximate unsupported terms. Differing `NonbondedForce` objects, including PME systems, are kept in full state-specific EVB evaluation; no local PME approximation is implemented.

## Next Step

The next recommended feature is native OPES on the EVB gap using a tabulated OpenMM bias. That is intentionally not implemented in this branch.
