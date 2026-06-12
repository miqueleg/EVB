# HG3.17 Q-region PMF and metadynamics reproduction

This workflow tests whether the calibrated HG3.17 Q-region EVB candidate can reproduce the previous full-state EVB gap PMF/metadynamics behavior and the user-provided g-xTB RC-to-TS barrier.

Selected candidate:

- `local_pme_q_atoms_cutoff_0.8`
- `delta_alpha_kj_mol: 405.2455501867761`
- `h12_kj_mol: 431.6758380801819`

The candidate uses Q-region EVB with local direct-space PME corrections. It is approximate relative to full endpoint-charge PME, and it is not OPES.

## Main Command

Quick check:

```bash
evb hg317-qregion-reproduce \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_reproduction_quick \
  --platform CUDA \
  --mode quick
```

Production setup without running long trajectories:

```bash
evb hg317-qregion-reproduce \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_reproduction \
  --platform CUDA \
  --mode production \
  --write-run-scripts-only
```

## Modes

- `quick`: 7 windows requested, 1000 umbrella/proxy steps, 1 metadynamics replica, 5000 metadynamics steps.
- `pilot`: 21 windows, 20000 umbrella steps, 2 replicas, 100000 metadynamics steps.
- `production`: 51 windows, 200000 umbrella steps, 3 replicas, 1000000 metadynamics steps.

All counts can be overridden with `--umbrella-steps`, `--metad-steps`, and `--replicas`.

## Outputs

```text
outputs/hg317_qregion_reproduction/
  configs/
  umbrella/windows/
  umbrella/analysis/pmf.csv
  umbrella/analysis/barrier_summary.json
  umbrella/analysis/qregion_gap_umbrella_pmf.png
  metad/rep01/colvar.csv
  metad/analysis/qregion_gap_metad_fel.csv
  metad/analysis/qregion_gap_metad_hist_fel.png
  metad/analysis/qregion_gap_metad_replicas_pmf.png
  comparison/barrier_comparison.csv
  comparison/barrier_comparison.md
  comparison/reproduction_report.md
  run_commands/
```

The quick umbrella route is an executable histogram fallback/proxy for Codex and CI checks. Production validation should use the generated long-run scripts and enough sampling for overlap/convergence.

## Barrier Criteria

Default reference target:

- g-xTB RC-to-TS barrier: 18.139814 kcal/mol
- warning threshold: 2 kcal/mol
- fail threshold: 3 kcal/mol

If previous full-state numeric PMF/metadynamics JSON or CSV outputs are available, the workflow reports them as legacy references. It does not extract numeric values from PNGs.

## Production Command

```bash
mkdir -p runs
bash scripts/run_hg317_qregion_reproduction/run_all_qregion_reproduction.sh \
  --output runs/hg317_qregion_reproduction_$(date +%Y%m%d_%H%M) \
  --platform CUDA \
  --mode production \
  --replicas 3 \
  --umbrella-steps 200000 \
  --metad-steps 1000000
```

Detached run:

```bash
RUN_DIR=runs/hg317_qregion_reproduction_$(date +%Y%m%d_%H%M)
mkdir -p "$RUN_DIR"
nohup bash scripts/run_hg317_qregion_reproduction/run_all_qregion_reproduction.sh \
  --output "$RUN_DIR" \
  --platform CUDA \
  --mode production \
  --replicas 3 \
  --umbrella-steps 200000 \
  --metad-steps 1000000 \
  > "$RUN_DIR/run.log" 2>&1 &

echo $! > "$RUN_DIR/pid.txt"
tail -f "$RUN_DIR/run.log"
```

Monitor and resume:

```bash
bash scripts/run_hg317_qregion_reproduction/monitor_reproduction.sh runs/hg317_qregion_reproduction_YYYYMMDD_HHMM
bash scripts/run_hg317_qregion_reproduction/resume_reproduction.sh --output runs/hg317_qregion_reproduction_YYYYMMDD_HHMM
```

## Cautions

This workflow is calibrated to the g-xTB profile and compares to previous full-state EVB numeric outputs when available. The selected candidate uses `local_pme_approx`, so it is not an exact full-state endpoint-charge PME reproduction. If the barrier differs by more than the configured thresholds, treat the model as not validated for the original full-state calculation.
