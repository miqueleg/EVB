# EVB Energy Decomposition Audit Report

## Summary

This audit fixes the exact EVB energy-decomposition implementation and its diagnostics. The implementation is mathematically equivalent for the tested systems, but the HG3.17 slowdown is a performance-design limitation, not a physics correctness failure: the two states have differing full-system PME `NonbondedForce` objects, so exact mode must still evaluate full state1 PME and full state2 PME in `e1`/`e2`.

The diagnostics now report that explicitly instead of hiding the cost behind small bonded-term counts.

## Git Commits

- Commit before this audit: `63a1ce8ff5745effd3b574566fe07d99e20c3167`
- Audit/fix commit: pending at report generation time; see final response for the committed SHA.

## Environment

- Python: `3.12.3`
- OpenMM: `8.5.2.dev-36a30cb`
- CUDA available: yes
- GPU: `NVIDIA GeForce RTX 5090`, driver `580.159.03`, memory `32607 MiB`
- CUDA benchmark precision: `{'Precision': 'mixed'}`

## What Changed

- Exact-mode config options are now honored by the builder:
  - `mode: legacy` uses the legacy full-state path.
  - `mode: exact` uses exact decomposition.
  - `fallback_to_legacy_for_unsupported_terms: true` keeps unsupported differing forces in full state-specific `e1`/`e2` and reports warnings.
  - `fallback_to_legacy_for_unsupported_terms: false` raises a clear `ValueError` when exact decomposition is not implemented.
  - `report: false` suppresses returned decomposition reports unless debug/benchmark code requests them.
- `CustomAngleForce` and `CustomTorsionForce` compatibility now use `getNumPerAngleParameters()` and `getNumPerTorsionParameters()` instead of bond-only APIs.
- `NonbondedForce` diagnostics now report particles, exceptions, method, full-system status, and duplicated full nonbonded status.
- Benchmark JSON summaries now include `duplicated_full_nonbonded` and decomposition partition diagnostics.
- Native gap metadynamics remains a separate OpenMM `BiasVariable` force. OpenMM does not expose a way to share the already-computed EVB `e1/e2` CVs from the lower-surface `CustomCVForce` with `Metadynamics`, so gap metadynamics evaluates state-energy CVs again for the bias.

## Tests

- Full test suite: `63 passed, 2 skipped`
- Toy CLI smoke tests completed:
  - `evb validate --config examples/toy_evb.yaml`
  - `evb singlepoint --config examples/toy_evb.yaml`
  - `evb minimize --config examples/toy_evb.yaml`
  - `evb sample-series --config examples/toy_evb.yaml`
  - `evb analyze --config examples/toy_evb.yaml`

New force-class coverage compares legacy vs exact decomposed EVB energy, shifted gap, forces, and minimization where applicable for:

- `HarmonicBondForce`
- `HarmonicAngleForce`
- `PeriodicTorsionForce`
- `RBTorsionForce`
- `CustomBondForce`
- `CustomAngleForce`
- `CustomTorsionForce`
- identical `NonbondedForce`
- differing PME `NonbondedForce` with fallback enabled
- differing PME `NonbondedForce` with fallback disabled

Small bonded/custom tests use `1e-6` CPU tolerances. PME force tests use a looser PME-only force tolerance because OpenMM CPU PME reductions differ at the `1e-4 kJ/mol/nm` level when the same force is evaluated through a different `CustomCVForce` tree; energies and gaps remain tight.

## Accuracy

HG3.17 same-coordinate CUDA mixed-precision comparison from `benchmarks/refactor_hg317_cuda_accuracy.json`:

| Quantity | Absolute difference |
| --- | ---: |
| EVB lower energy | `0.000234 kJ/mol` |
| Shifted gap | `0.000112 kJ/mol` |
| Force RMSD | `3.52e-05 kJ/mol/nm` |
| Force max abs | `2.99e-04 kJ/mol/nm` |

These differences are CUDA mixed-precision reduction-level differences on a 32,777-atom PME system. The deterministic small-system CPU tests enforce tighter equivalence.

## Toy Benchmarks

| Run | Context s | MD wall s | steps/s | ns/day | speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU legacy | 0.00154 | 0.375 | 2847.3 | 246.0 | 1.00x |
| CPU decomposed | 0.00200 | 0.468 | 2296.0 | 198.4 | 0.81x |
| CUDA legacy | 0.346 | 1.160 | 4453.3 | 384.8 | 1.00x |
| CUDA decomposed | 0.341 | 1.277 | 6387.8 | 551.9 | 1.43x |

The toy system is not representative of production systems. It has no common terms to save, and CUDA timings are noisy at this size.

## HG3.17 Status

Prepared HG3.17 EVB-ready OpenMM files were available locally:

- `prep/hg317_full_irc/evb_ready/state1_evb_ready/system.xml`
- `prep/hg317_full_irc/evb_ready/state2_evb_ready/system.xml`
- `prep/hg317_full_irc/evb_ready/state1_evb_ready/coordinates.pdb`
- `prep/hg317_full_irc/evb_ready/state2_evb_ready/coordinates.pdb`

Short native OpenMM gap metadynamics benchmarks were run with `examples/hg317_evb_gap_metad.yaml`, `--workflow gap-metad`, `--steps 2000`, `--repeats 3`, `--platform CUDA`, and `--no-forces`.

| Run | Context s | MD wall s | steps/s | ns/day | speed ratio | MD time ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HG3.17 legacy | 1.829 | 2.406 | 831.2 | 17.95 | 1.00x | 1.00x |
| HG3.17 decomposed | 7.826 | 2.623 | 762.6 | 16.47 | 0.92x | 1.09x longer |

For computing time rather than steps/s: decomposed HG3.17 metadynamics took `2.623 s` of MD wall time versus `2.406 s` legacy for the same 2000 steps. That is about `9%` slower during stepping. Including context creation, the decomposed run is much worse because context creation rises from `1.829 s` to `7.826 s`.

## Decomposition Diagnostics

HG3.17 decomposed mode reports:

| Partition | Forces by class | Nonbonded particles | Nonbonded exceptions | Method | Full-system nonbonded |
| --- | --- | ---: | ---: | --- | --- |
| common | `HarmonicBondForce: 1`, `HarmonicAngleForce: 1`, `PeriodicTorsionForce: 1` | 0 | 0 | none | no |
| state1 | `HarmonicBondForce: 1`, `HarmonicAngleForce: 1`, `PeriodicTorsionForce: 1`, `NonbondedForce: 1` | 32777 | 53093 | PME | yes |
| state2 | `HarmonicBondForce: 1`, `HarmonicAngleForce: 1`, `PeriodicTorsionForce: 1`, `NonbondedForce: 1` | 32777 | 53090 | PME | yes |

- `duplicated_full_nonbonded`: `true`
- `n_common_terms`: `33552`
- `n_state1_terms + n_state2_terms`: `171779`

Warnings:

```text
Exact decomposition still evaluates full state-specific NonbondedForce objects for both states; PME/nonbonded cost is duplicated and speedup should not be expected.
NonbondedForce differs between states and cannot be decomposed exactly in PME/local nonbonded form; full state-specific NonbondedForce objects are required unless fallback is disabled.
```

## Scientific Caution

Exact mode preserves the EVB physics and does not approximate PME. If a full PME `NonbondedForce` differs between states, exact decomposed mode keeps full state-specific PME forces when fallback is enabled. That is correct, but it does not implement the intended production speed strategy and can be slower because the force tree is larger.

The current exact decomposition is therefore a correctness and architecture improvement, not a speedup for solvated enzyme systems whose dominant PME force differs between diabatic states.

## Next Step

The next performance step is either:

1. Restructure prepared EVB systems so a truly identical nonbonded baseline can be shared exactly and only exact residual terms remain state-specific.
2. Add an explicitly experimental `local_nonbonded_approx` mode with direct-space reactive-region corrections, disabled by default, with measured energy/gap/force errors versus legacy.

Native OPES or further metadynamics features should remain a later branch.
