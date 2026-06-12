# HG3.17 Q-region PMF and metadynamics reproduction report

## Summary

This branch adds a runnable workflow for checking whether the calibrated HG3.17 Q-region EVB candidate reproduces previous EVB-gap PMF/metadynamics behavior and the g-xTB RC-to-TS barrier. It generates Q-region umbrella and native table-metadynamics configs, long-run shell scripts, quick/pilot/production modes, analysis outputs, barrier comparisons, and speed summaries.

No OPES implementation was added.

## Git SHA

Pre-commit base SHA during report generation: `33552e98f1e6902c788b5f9deaf4c239ffa39979`

## Selected Candidate

- candidate: `local_pme_q_atoms_cutoff_0.8`
- `delta_alpha_kj_mol`: 405.2455501867761
- `h12_kj_mol`: 431.6758380801819
- nonbonded model: `local_pme_approx`
- duplicated full nonbonded: false in quick Q-region runs
- approximation used: true

## Tests Run

- `.venv/bin/python -m compileall -q src/kemp_evb tests scripts/hg317_qregion_reproduce_pmf_metad.py`
- `.venv/bin/python -m pytest tests/test_hg317_reproduction_workflow.py -q` -> 7 passed
- `.venv/bin/python -m pytest -q` -> 114 passed, 2 skipped

## Quick Workflow Run

Command:

```bash
.venv/bin/evb hg317-qregion-reproduce \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output outputs/hg317_qregion_reproduction_quick \
  --platform CUDA \
  --mode quick
```

Quick mode settings: 7 umbrella windows requested, 1000 umbrella/proxy steps, 1 native table-metad replica, 5000 metadynamics steps.

The quick umbrella route is a short executable histogram fallback/proxy. It is not a converged production WHAM/MBAR PMF.

## Barrier Table

| method | barrier kcal/mol | delta vs g-xTB | delta vs legacy | status |
| --- | ---: | ---: | ---: | --- |
| g-xTB RC->TS | 18.13981382 | 0.0 | None | reference |
| q_region umbrella | 0.0 | -18.13981382 | None | fail |
| q_region table-metad | 0.0 | -18.13981382 | None | fail |

The quick barriers fail the default 3 kcal/mol threshold, which is expected for the very short quick run and is intentionally reported as failed validation rather than claimed as a PMF result.

## Speed Table

| method | steps/s | ns/day | speedup vs legacy |
| --- | ---: | ---: | ---: |
| q_region umbrella | 2203.592298480985 | 47.59759364718927 | n/a |
| q_region table-metad | 2223.0066818501846 | 48.01694432796399 | n/a |

Previous full-state numeric reference summaries were not found in JSON/CSV form, so no speedup ratio to legacy was computed. PNGs were not parsed.

## Generated Outputs

- `outputs/hg317_qregion_reproduction_quick/configs/qregion_gap_umbrella.yaml`
- `outputs/hg317_qregion_reproduction_quick/configs/qregion_gap_table_metad.yaml`
- `outputs/hg317_qregion_reproduction_quick/umbrella/analysis/pmf.csv`
- `outputs/hg317_qregion_reproduction_quick/umbrella/analysis/qregion_gap_umbrella_pmf.png`
- `outputs/hg317_qregion_reproduction_quick/metad/analysis/qregion_gap_metad_fel.csv`
- `outputs/hg317_qregion_reproduction_quick/metad/analysis/qregion_gap_metad_hist_fel.png`
- `outputs/hg317_qregion_reproduction_quick/metad/analysis/qregion_gap_metad_replicas_pmf.png`
- `outputs/hg317_qregion_reproduction_quick/comparison/barrier_comparison.csv`
- `outputs/hg317_qregion_reproduction_quick/comparison/barrier_comparison.md`
- `outputs/hg317_qregion_reproduction_quick/comparison/reproduction_report.md`
- `outputs/hg317_qregion_reproduction_quick/run_commands/*.sh`
- `scripts/run_hg317_qregion_reproduction/*.sh`

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

Detached command:

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

Monitor/resume:

```bash
bash scripts/run_hg317_qregion_reproduction/monitor_reproduction.sh runs/hg317_qregion_reproduction_YYYYMMDD_HHMM
bash scripts/run_hg317_qregion_reproduction/resume_reproduction.sh --output runs/hg317_qregion_reproduction_YYYYMMDD_HHMM
```

## Scientific Interpretation

The selected Q-region model is calibrated to the g-xTB profile and uses approximate local PME corrections. It is not an exact full-state endpoint-charge PME Hamiltonian. The workflow will mark Q-region PMF/metadynamics results as warning/fail when the barrier differs from g-xTB or legacy numeric references by the configured thresholds. Default thresholds are 2 kcal/mol warning and 3 kcal/mol fail.

The next scientific step is to run the production scripts outside Codex, inspect overlap/convergence and replica agreement, and only then decide whether the calibrated Q-region model is validated enough for `feature/native-opes-gap`.
