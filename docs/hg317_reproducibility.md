# HG3.17 EVB reproducibility notes

This repository includes the code, configuration files, and lightweight scripts
needed to reproduce the HG3.17 EVB workflow developed during testing. Large
generated files are intentionally excluded from git: trajectories, PMF output
directories, OpenMM runtime logs, PLUMED kernels, and generated EVB-ready system
bundles.

## Included inputs and examples

- `examples/HD3.17_IRC.xyz`: cluster-model IRC-like geometry path. This is
  geometry only; XYZ comments are not treated as thermodynamic energies.
- `examples/hg317_irc_mapping_barrier_calibrated.yaml`: IRC setup and barrier
  calibration template.
- `examples/hg317_evb_gap_metad.yaml`: native OpenMM metadynamics on the shifted
  EVB gap.
- `examples/hg317_evb_plumed_metad.yaml` and
  `examples/hg317_evb_plumed_opes.yaml`: PLUMED geometrical-CV templates.
- `runs/hg317_evb_enhanced_sampling/`: portable 2-GPU launcher scripts.

The generated OpenMM bundles under `prep/**/evb_ready/` are not committed. They
are system-specific outputs and should be regenerated from the user's own
prepared diabatic states.

## Environment

For native OpenMM EVB and gap metadynamics:

```bash
mamba env create -f environment.yml
mamba activate openmm-evb
pip install -e ".[openmm,analysis,test]"
```

For PLUMED geometrical-CV examples:

```bash
mamba env create -f environment-plumed.yml
mamba activate openmm-evb-plumed
pip install -e ".[openmm,analysis,plumed,test]"
python scripts/check_openmm_plumed_cuda.py --require-cuda
```

## Regenerate EVB-ready inputs

Start from compatible AMBER or OpenMM diabatic state files. The original
endpoint files are not modified.

```bash
evb prepare-evb-inputs \
  --config examples/hg317_irc_mapping_barrier_calibrated.yaml \
  --output prep/hg317_full_irc/evb_ready
```

Then run the IRC setup/calibration with the generated YAML:

```bash
evb setup-from-irc \
  --config prep/hg317_full_irc/evb_ready/hg317_irc_mapping_barrier_calibrated_evb_ready.yaml \
  --write-window-config
```

Inspect `analysis/evb_reference_fit_from_irc.json` and
`analysis/irc_diabatic_scan.csv` before launching expensive sampling.

## Run native gap metadynamics on two GPUs

The launcher uses the active conda environment by default. Override with
`ENV_PY=/path/to/python` if needed.

```bash
bash runs/hg317_evb_enhanced_sampling/run_gap_metad_2gpu.sh
```

The script assigns one walker to each GPU with `CUDA_VISIBLE_DEVICES=0` and
`CUDA_VISIBLE_DEVICES=1`. If previous PMF seed structures exist locally, it uses
them as starting coordinates; otherwise it starts from the coordinates in the
example config.

## Analyze and visualize

```bash
python scripts/plot_gap_metad_results.py \
  --base outputs/hg317_evb_gap_metad \
  --out outputs/hg317_evb_gap_metad/summary

python scripts/extract_gap_pseudo_trajectory.py \
  --base outputs/hg317_evb_gap_metad \
  --bins 61
```

The pseudo-trajectory is written as a multi-model PDB ordered from negative to
positive shifted EVB gap. It is for visualization only, not a physical
time-ordered trajectory.

## Scientific caution

The cluster-model IRC and the pseudo-trajectory are not EVB PMFs. The final
activation free energy must come from EVB sampling over the energy-gap
coordinate with adequate overlap, convergence checks, and independent
replicates.
