# Exact EVB Energy Decomposition

The native OpenMM EVB surface can optionally evaluate the common part of both diabatic states only once:

```text
E1 = E_common + e1
E2 = E_common + e2
E_EVB = E_common + 0.5*(e1 + e2 + delta_alpha) - sqrt(0.25*(e1 - e2 - delta_alpha)^2 + H12^2)
gap = e1 - e2 - delta_alpha
```

This is useful because protein, solvent, and other unchanged force terms can be identical in both diabatic states. Evaluating those terms once reduces duplicated work without changing the physics.

## Exact Mode

Enable the mode in modern YAML configs:

```yaml
evb:
  energy_decomposition:
    enabled: true
    mode: exact
    fallback_to_legacy_for_unsupported_terms: true
    report: true
```

Existing configs without this block continue to use the legacy full-state EVB path.

## What Is Decomposed

The implementation first compares whole OpenMM forces by serialized XML. Identical forces are moved to `E_common`.

For differing forces, exact per-term decomposition is implemented for:

- `HarmonicBondForce`
- `HarmonicAngleForce`
- `PeriodicTorsionForce`
- `RBTorsionForce`
- `CustomBondForce` when energy expression and parameter definitions match
- `CustomAngleForce` when energy expression and parameter definitions match
- `CustomTorsionForce` when energy expression and parameter definitions match

Identical terms go to `E_common`; terms that differ remain in `e1` and `e2`.

## Unsupported Terms

If a differing force cannot be decomposed exactly, it remains in the state-specific path and is reported. This preserves exactness by default.

`NonbondedForce` is only decomposed when the full force is identical between states. Differing `NonbondedForce` objects, including PME systems, are kept as full state-specific EVB contributions. No local PME approximation is used.

## Automatic Adiabatic Setup

For workflows that already have state1/state2 AMBER inputs plus an IRC, the helper command can create EVB-ready OpenMM bundles and then run IRC-based calibration/setup in one step:

```bash
evb prepare-adiabatic-system \
  --config examples/hg317_irc_mapping_barrier_calibrated.yaml \
  --output prep/hg317_full_irc/evb_ready \
  --write-window-config
```

The command runs the same conservative preparation used by `prepare-evb-inputs`, writes all derived files, and then calls `setup-from-irc`. If an alpha-carbon mapping is missing and the config has a 5RGE-style PDB plus fixed IRC atoms, it derives the mapping and records it in the configured mapping path.

This does not replace full system parameterization from raw structures. For HG3.17/5RGE, `evb prepare-hg317-system --config examples/hg317_system_prep.yaml` scaffolds that earlier stage. It requires AmberTools (`antechamber`, `parmchk2`, `tleap`) when execution is requested, supports explicit protonation and water-retention choices, allows user-supplied ligand files, writes intermediates, and fails loudly if AM1-BCC/GAFF2/tleap cannot be run.

## Benchmarking

Run the benchmark script with legacy and decomposed modes:

```bash
python scripts/benchmark_evb_decomposition.py --config examples/toy_evb.yaml --platform CUDA --steps 5000 --repeats 3 --mode legacy --output benchmarks/refactor_toy_cuda_legacy.json --require-inputs true
python scripts/benchmark_evb_decomposition.py --config examples/toy_evb.yaml --platform CUDA --steps 5000 --repeats 3 --mode decomposed --output benchmarks/refactor_toy_cuda_decomposed.json --require-inputs true
```

The script also supports workflow-specific timing for native gap umbrella and native OpenMM gap metadynamics:

```bash
python scripts/benchmark_evb_decomposition.py --config benchmarks/hg317_gap_umbrella_benchmark.yaml --platform CUDA --workflow gap-umbrella --mode decomposed --steps 2000 --repeats 3 --output benchmarks/hg317_gap_umbrella_cuda_decomposed.json --require-inputs true --no-forces
python scripts/benchmark_evb_decomposition.py --config benchmarks/hg317_gap_metad_benchmark.yaml --platform CUDA --workflow gap-metad --mode decomposed --steps 2000 --repeats 3 --output benchmarks/hg317_gap_metad_cuda_decomposed.json --require-inputs true --no-forces
```

CUDA benchmarks default to `--cuda-precision mixed`; use `--cuda-precision single` or `--cuda-precision double` to override. The script records CUDA usage, platform properties, timing, mean EVB energy, shifted gap, optional force norms, decomposition counts, warnings, git SHA, and timestamp.
