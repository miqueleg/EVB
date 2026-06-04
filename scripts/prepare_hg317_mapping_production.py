from __future__ import annotations

import json
from pathlib import Path


N_REPLICATES = 30
SEED_BASE = 417000
N_WINDOWS = 41
TIMESTEP_FS = 0.5
EQUILIBRATION_PS = 5.0
PRODUCTION_PS = 200.0
REPORT_STRIDE_PS = 0.5


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "runs" / "hg317_evb_mapping"
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for replica_index in range(1, N_REPLICATES + 1):
        (config_dir / f"rep{replica_index:02d}.yaml").write_text(_render_config(replica_index), encoding="utf-8")

    manifest = {
        "description": "HG3.17 constant-H12 EVB mapped-Hamiltonian production, Q6-like first production route.",
        "requires_platform": "CUDA",
        "recommended_preflight": "PYTHONPATH=src python scripts/run_hg317_mapping_timestep_ladder.py",
        "n_replicates": N_REPLICATES,
        "n_windows": N_WINDOWS,
        "lambda_values": _lambda_values(),
        "timestep_fs": TIMESTEP_FS,
        "equilibration_ps_per_window": EQUILIBRATION_PS,
        "production_ps_per_window": PRODUCTION_PS,
        "total_production_ns": N_REPLICATES * N_WINDOWS * PRODUCTION_PS * 1.0e-3,
        "delta_alpha_kj_mol": -53328.25448088075,
        "h12_kj_mol": 49606.147558391305,
        "q6_like_defaults": [
            "mapping windows before direct gap umbrellas",
            "CUDA required",
            "far-field positional restraints outside a 1.2 nm active-site/substrate radius",
            "reactive donor/proton/acceptor atoms excluded when EVB wrapper constraints are copied",
            "independent replica seeds",
            "gap logged for reweighting/PMF reconstruction",
        ],
        "notes": [
            "Mapped timestep ladder with seed relaxation showed 0.5 fs stable for w000/w020/w040; 1.0 fs failed for w020.",
            "Raw TS seeds are not forced into production by default; windows chain from equilibrated neighbors and optional seed relaxation handles explicit seeds.",
            "Use gap umbrellas only after mapped-window overlap identifies poorly sampled regions.",
            "Confirm the OpenMM bundle paths correspond to HG3.17 before interpreting enzyme-specific results.",
        ],
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    launcher = run_root / "run_all_cuda_2gpu.sh"
    launcher.write_text(_render_launcher(), encoding="utf-8")
    launcher.chmod(0o755)
    summary = run_root / "summarize.sh"
    summary.write_text(_render_summary_script(), encoding="utf-8")
    summary.chmod(0o755)

    print(f"Wrote {N_REPLICATES} mapped CUDA replica configs to {config_dir}")
    print(f"Launcher: {launcher}")
    print(f"Manifest: {run_root / 'manifest.json'}")


def _lambda_values() -> list[float]:
    return [round(index / (N_WINDOWS - 1), 6) for index in range(N_WINDOWS)]


def _steps(ps: float) -> int:
    return int(round(ps * 1000.0 / TIMESTEP_FS))


def _render_config(replica_index: int) -> str:
    seed = SEED_BASE + replica_index
    lambdas = ", ".join(f"{value:.6g}" for value in _lambda_values())
    output_dir = f"outputs/hg317_evb_mapping_replicates/rep{replica_index:02d}"
    return f"""project:
  name: hg317-evb-mapping-rep{replica_index:02d}
  output_dir: {output_dir}
reaction:
  metadata:
    name: hg317_evb_mapping_rep{replica_index:02d}
    phase: enzyme
    temperature_k: 300.0
    pressure_bar: 1.0
    notes: >
      CUDA production HG3.17 EVB mapped-Hamiltonian replicate. Mapping is the first production route;
      native gap umbrellas are reserved for follow-up refinement.
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
  mode: mapping
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
  seed_relaxation:
    enabled: true
    minimization_steps: 500
    equilibration_steps: 0
    restraint_force_constant_kj_mol_nm2: 250.0
    restraint_decay: [1.0, 0.5, 0.1, 0.0]
  windows:
    mapping:
      lambda_values: [{lambdas}]
  seed_windows:
    - window_id: w000
      coordinates: prep/kemp_qm_openmm/07_openmm_bundles/RC/coordinates.pdb
    - window_id: w040
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
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print("OpenMM platforms:", ", ".join(names))
print("OpenMM version:", mm.version.version)
if "CUDA" not in names:
    raise SystemExit("CUDA platform is required for this production run.")
for device in ("0", "1"):
    system = mm.System()
    system.addParticle(1.0 * unit.amu)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    try:
        context = mm.Context(system, integrator, platform, {"DeviceIndex": device})
    except Exception as exc:
        raise SystemExit(f"CUDA device {device} preflight failed: {exc}")
    del context, integrator
    print(f"CUDA device {device} preflight: ok")
PY

mkdir -p runs/hg317_evb_mapping/logs

run_replica() {
  local cfg="$1"
  local gpu="$2"
  local rep
  rep="$(basename "$cfg" .yaml)"
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[$(date -Is)] GPU $gpu validate $rep"
  python -m evb.cli validate --config "$cfg" > "runs/hg317_evb_mapping/logs/${rep}_validate.log" 2>&1
  echo "[$(date -Is)] GPU $gpu sample-series $rep"
  python -m evb.cli sample-series --config "$cfg" > "runs/hg317_evb_mapping/logs/${rep}_sample.log" 2>&1
  echo "[$(date -Is)] GPU $gpu analyze $rep"
  python -m evb.cli analyze --config "$cfg" > "runs/hg317_evb_mapping/logs/${rep}_analyze.log" 2>&1
}

index=0
for cfg in runs/hg317_evb_mapping/configs/rep*.yaml; do
  gpu=$(( index % 2 ))
  run_replica "$cfg" "$gpu" &
  index=$(( index + 1 ))
  if (( index % 2 == 0 )); then
    wait
  fi
done
wait
"""


def _render_summary_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=src

python scripts/summarize_hg317_mapping_replicates.py
"""


if __name__ == "__main__":
    main()
