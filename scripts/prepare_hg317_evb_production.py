from __future__ import annotations

import json
from pathlib import Path


N_REPLICATES = 30
SEED_BASE = 317000
WINDOW_CENTERS_KJ_MOL = [float(20000 + 5000 * index) for index in range(49)]
FORCE_CONSTANT_KJ_MOL2 = 1.0e-7
TIMESTEP_FS = 0.1
EQUILIBRATION_PS = 5.0
PRODUCTION_PS = 200.0
REPORT_STRIDE_PS = 0.5


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "runs" / "hg317_evb_gap_umbrella"
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for replica_index in range(1, N_REPLICATES + 1):
        (config_dir / f"rep{replica_index:02d}.yaml").write_text(_render_config(replica_index), encoding="utf-8")

    manifest = {
        "description": "HG3.17 constant-H12 EVB production gap umbrellas calibrated to g-xTB RC/TS/PROD targets.",
        "requires_platform": "CUDA",
        "n_replicates": N_REPLICATES,
        "n_windows": len(WINDOW_CENTERS_KJ_MOL),
        "window_centers_kj_mol": WINDOW_CENTERS_KJ_MOL,
        "force_constant_kj_mol2": FORCE_CONSTANT_KJ_MOL2,
        "timestep_fs": TIMESTEP_FS,
        "equilibration_ps_per_window": EQUILIBRATION_PS,
        "production_ps_per_window": PRODUCTION_PS,
        "total_production_ns": N_REPLICATES * len(WINDOW_CENTERS_KJ_MOL) * PRODUCTION_PS * 1.0e-3,
        "delta_alpha_kj_mol": -53328.25448088075,
        "h12_kj_mol": 49606.147558391305,
        "target_barrier_kj_mol": 75.897781,
        "target_reaction_free_energy_kj_mol": -156.280437,
        "notes": [
            "Use gap PMF overlap/blocking/replicate summaries before interpreting the barrier.",
            "The available OpenMM bundle paths are not explicitly named HG3.17; confirm they are the intended HG3.17 diabatic states.",
            "Default protocol uses far-field restraints, reactive-atom constraint exclusion, and umbrella-ramp equilibration.",
            "Timestep ladder showed 0.1 fs stable for u000/u019/u038/u048; 0.2 fs failed for u048.",
        ],
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    launcher = run_root / "run_all_cuda.sh"
    launcher.write_text(_render_launcher(), encoding="utf-8")
    launcher.chmod(0o755)
    summary = run_root / "summarize.sh"
    summary.write_text(_render_summary_script(), encoding="utf-8")
    summary.chmod(0o755)

    print(f"Wrote {N_REPLICATES} CUDA replica configs to {config_dir}")
    print(f"Launcher: {launcher}")
    print(f"Manifest: {run_root / 'manifest.json'}")


def _steps(ps: float) -> int:
    return int(round(ps * 1000.0 / TIMESTEP_FS))


def _render_config(replica_index: int) -> str:
    seed = SEED_BASE + replica_index
    centers = ", ".join(f"{center:.1f}" for center in WINDOW_CENTERS_KJ_MOL)
    output_dir = f"outputs/hg317_evb_gap_umbrella_replicates/rep{replica_index:02d}"
    return f"""project:
  name: hg317-evb-gap-rep{replica_index:02d}
  output_dir: {output_dir}
reaction:
  metadata:
    name: hg317_evb_gap_rep{replica_index:02d}
    phase: enzyme
    temperature_k: 300.0
    pressure_bar: 1.0
    notes: >
      CUDA production HG3.17 EVB gap umbrella replicate. Constant H12 fitted to g-xTB RC/TS/PROD targets.
  atoms:
    donor: 6
    proton: 15
    acceptor: 19
  substrate_atoms: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
states:
  state1:
    topology: prep/kemp_qm_openmm/07_openmm_bundles/RC/system.xml
    coordinates: prep/kemp_qm_openmm/07_openmm_bundles/RC/coordinates.pdb
    format: openmm
  state2:
    topology: prep/kemp_qm_openmm/07_openmm_bundles/PROD/system.xml
    coordinates: prep/kemp_qm_openmm/07_openmm_bundles/PROD/coordinates.pdb
    format: openmm
evb:
  coupling_model:
    model: constant
    parameters:
      delta_alpha_kj_mol: -53328.25448088075
      h12_kj_mol: 49606.147558391305
sampling:
  mode: gap_umbrella
  integrator:
    name: LangevinMiddle
    timestep_fs: {TIMESTEP_FS}
    friction_per_ps: 1.0
    seed: {seed}
  md:
    equilibration_steps: {_steps(EQUILIBRATION_PS)}
    production_steps: {_steps(PRODUCTION_PS)}
    report_stride: {_steps(REPORT_STRIDE_PS)}
    save_stride: null
    platform: CUDA
    temperature_k: 300.0
    nonbonded_method: PME
    constraints: None
    minimize_steps: 500
    minimize_tolerance: 10.0
  far_field_restraint:
    enabled: true
    radius_nm: 1.2
    force_constant_kj_mol_nm2: 25.0
  umbrella_ramp:
    enabled: true
    fractions: [0.05, 0.1, 0.25, 0.5, 1.0]
  seed_relaxation:
    enabled: true
    minimization_steps: 500
    equilibration_steps: 0
    restraint_force_constant_kj_mol_nm2: 250.0
    restraint_decay: [1.0, 0.5, 0.1, 0.0]
  windows:
    gap_umbrella:
      centers_kj_mol: [{centers}]
      force_constant_kj_mol2: {FORCE_CONSTANT_KJ_MOL2}
  seed_windows:
    - window_id: u000
      coordinates: prep/kemp_qm_openmm/07_openmm_bundles/RC/coordinates.pdb
    - window_id: u019
      coordinates: prep/kemp_qm_openmm/05_templates/TS_solvated_template.pdb
    - window_id: u038
      coordinates: prep/kemp_qm_openmm/07_openmm_bundles/PROD/coordinates.pdb
    - window_id: u048
      coordinates: prep/kemp_qm_openmm/07_openmm_bundles/PROD/coordinates.pdb
observables:
  gap:
    shifted: true
  reaction_coordinates:
    - name: proton_transfer_rc
      kind: difference_of_distances
      atom1: 6
      atom2: 15
      atom3: 19
      event_threshold_nm: 0.0
    - name: ring_opening_rc
      kind: distance
      atom1: 1
      atom2: 11
  distances:
    - name: donor_h
      atom1: 6
      atom2: 15
    - name: h_acceptor
      atom1: 15
      atom2: 19
analysis:
  histogram:
    bin_min_kj_mol: 0.0
    bin_max_kj_mol: 275000.0
    n_bins: 550
  pmf:
    temperature_k: 300.0
    zero_mode: reactant_min
  barrier:
    reactant_region: [15000.0, 35000.0]
    product_region: [240000.0, 270000.0]
  uncertainty:
    blocks: 5
    bootstrap_samples: 200
"""


def _render_launcher() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=src

python - <<'PY'
import openmm as mm
from openmm import unit
try:
    import importlib.metadata as metadata
except ImportError:
    metadata = None
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print("OpenMM platforms:", ", ".join(names))
print("OpenMM version:", mm.version.version)
if metadata is not None:
    for package in ("cuda-nvrtc", "cuda-version"):
        try:
            print(f"{package}:", metadata.version(package))
        except metadata.PackageNotFoundError:
            pass
if "CUDA" not in names:
    raise SystemExit("CUDA platform is required for this production run.")
system = mm.System()
system.addParticle(1.0 * unit.amu)
integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
platform = mm.Platform.getPlatformByName("CUDA")
try:
    context = mm.Context(system, integrator, platform)
except Exception as exc:
    raise SystemExit(
        "CUDA platform is installed but no usable CUDA context is available. "
        "If the error is CUDA_ERROR_UNSUPPORTED_PTX_VERSION, the NVIDIA driver is too old "
        "for the CUDA/NVRTC runtime in this conda environment. "
        f"Original error: {exc}"
    )
del context, integrator
print("CUDA context preflight: ok")
PY

mkdir -p runs/hg317_evb_gap_umbrella/logs

for cfg in runs/hg317_evb_gap_umbrella/configs/rep*.yaml; do
  rep="$(basename "$cfg" .yaml)"
  echo "[$(date -Is)] validate $rep"
  python -m evb.cli validate --config "$cfg" > "runs/hg317_evb_gap_umbrella/logs/${rep}_validate.log" 2>&1
  echo "[$(date -Is)] sample-series $rep"
  python -m evb.cli sample-series --config "$cfg" > "runs/hg317_evb_gap_umbrella/logs/${rep}_sample.log" 2>&1
  echo "[$(date -Is)] analyze $rep"
  python -m evb.cli analyze --config "$cfg" > "runs/hg317_evb_gap_umbrella/logs/${rep}_analyze.log" 2>&1
done
"""


def _render_summary_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=src

python scripts/summarize_hg317_evb_replicates.py
"""


if __name__ == "__main__":
    main()
