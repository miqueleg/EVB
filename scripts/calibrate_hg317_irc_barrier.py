from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from kemp_evb.evb import EVBHamiltonian, EVBParameters


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hg317_irc_evb"
RUN_ROOT = ROOT / "runs" / "hg317_irc_mapping_barrier_calibrated"
EXAMPLE_CONFIG = ROOT / "examples" / "hg317_irc_mapping_barrier_calibrated.yaml"

KJ_TO_KCAL = 1.0 / 4.184

# Barrier-constrained constant-H12 calibration from the matched-state IRC single points.
# This pair exactly matches the g-xTB IRC barrier from examples/HG3.17_CM_IRC.xyz,
# but it does not reproduce the full IRC shape or reaction energy.
DELTA_ALPHA_KJ_MOL = 75.0
H12_KJ_MOL = 2393.447482577291

N_REPLICATES = 30
N_WINDOWS = 41
SEED_BASE = 531700
TIMESTEP_FS = 0.5
EQUILIBRATION_PS = 5.0
PRODUCTION_PS = 200.0
REPORT_STRIDE_PS = 0.5

DONOR_ATOM = 4548
PROTON_ATOM = 4555
ACCEPTOR_ATOM = 1954
SUBSTRATE_ATOMS = list(range(4540, 4556))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _read_singlepoints()
    _apply_barrier_calibration(rows)
    metrics = _metrics(rows)
    _write_barrier_csv(rows)
    _write_plots(rows)
    _write_reports(metrics)
    _write_production_configs(metrics)
    print(json.dumps(metrics, indent=2))


def _read_singlepoints() -> list[dict]:
    path = OUT / "irc_singlepoints.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/run_hg317_irc_evb.py first; missing {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {"frame": int(row["frame"])}
            for key, value in row.items():
                if key == "frame":
                    continue
                parsed[key] = None if value in {"", "None"} else float(value)
            rows.append(parsed)
    return rows


def _apply_barrier_calibration(rows: list[dict]) -> None:
    hamiltonian = EVBHamiltonian(EVBParameters(delta_alpha=DELTA_ALPHA_KJ_MOL, h12=H12_KJ_MOL))
    evb0 = None
    for row in rows:
        e1 = row["e1_kj_mol"]
        e2 = row["e2_kj_mol"]
        evb, w1, w2 = hamiltonian.lower_eigenvalue(e1, e2)
        row["barrier_calibrated_gap_kj_mol"] = hamiltonian.gap(e1, e2)
        row["barrier_calibrated_evb_kj_mol"] = evb
        row["barrier_calibrated_weight1"] = w1
        row["barrier_calibrated_weight2"] = w2
        if evb0 is None and row["qm_relative_kj_mol"] is not None:
            evb0 = evb
    if evb0 is None:
        raise ValueError("No finite QM reference frame found.")
    for row in rows:
        row["barrier_calibrated_evb_relative_kj_mol"] = row["barrier_calibrated_evb_kj_mol"] - evb0
        for key in [
            "qm_relative",
            "barrier_calibrated_gap",
            "barrier_calibrated_evb",
            "barrier_calibrated_evb_relative",
        ]:
            kj_key = f"{key}_kj_mol"
            if kj_key in row:
                row[f"{key}_kcal_mol"] = None if row[kj_key] is None else row[kj_key] * KJ_TO_KCAL


def _metrics(rows: list[dict]) -> dict:
    finite = [row for row in rows if row["qm_relative_kj_mol"] is not None]
    qm = np.asarray([row["qm_relative_kcal_mol"] for row in finite], dtype=float)
    evb = np.asarray([row["barrier_calibrated_evb_relative_kcal_mol"] for row in finite], dtype=float)
    residual = evb - qm
    worst = int(np.argmax(np.abs(residual)))
    return {
        "status": "barrier_calibrated",
        "calibration_scope": "constant-H12 EVB parameters constrained to match the g-xTB IRC barrier only",
        "delta_alpha_kj_mol": DELTA_ALPHA_KJ_MOL,
        "h12_kj_mol": H12_KJ_MOL,
        "delta_alpha_kcal_mol": DELTA_ALPHA_KJ_MOL * KJ_TO_KCAL,
        "h12_kcal_mol": H12_KJ_MOL * KJ_TO_KCAL,
        "qm_barrier_kcal_mol": float(np.nanmax(qm) - qm[0]),
        "evb_barrier_kcal_mol": float(np.nanmax(evb) - evb[0]),
        "qm_reaction_energy_kcal_mol": float(qm[-1] - qm[0]),
        "evb_reaction_energy_kcal_mol": float(evb[-1] - evb[0]),
        "rmse_kcal_mol": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_residual_kcal_mol": float(np.max(np.abs(residual))),
        "worst_residual_frame": finite[worst]["frame"],
        "warning": (
            "This calibration matches the barrier, but the constant-H12 model with the current diabatic states "
            "does not reproduce the full IRC shape or reaction energy. Treat the PMF run as a sampling test until "
            "the diabatic state parameters are recalibrated."
        ),
    }


def _write_barrier_csv(rows: list[dict]) -> None:
    fields = [
        "frame",
        "qm_energy_hartree",
        "qm_relative_kcal_mol",
        "barrier_calibrated_evb_relative_kcal_mol",
        "barrier_calibrated_gap_kcal_mol",
        "barrier_calibrated_weight1",
        "barrier_calibrated_weight2",
        "qm_relative_kj_mol",
        "barrier_calibrated_evb_relative_kj_mol",
        "barrier_calibrated_gap_kj_mol",
    ]
    with (OUT / "irc_evb_barrier_calibrated_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_plots(rows: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = np.asarray([row["frame"] for row in rows], dtype=float)
    qm = np.asarray([np.nan if row["qm_relative_kcal_mol"] is None else row["qm_relative_kcal_mol"] for row in rows], dtype=float)
    evb = np.asarray([row["barrier_calibrated_evb_relative_kcal_mol"] for row in rows], dtype=float)
    gap = np.asarray([row["barrier_calibrated_gap_kcal_mol"] for row in rows], dtype=float)
    w1 = np.asarray([row["barrier_calibrated_weight1"] for row in rows], dtype=float)
    w2 = np.asarray([row["barrier_calibrated_weight2"] for row in rows], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(x, qm, label="g-xTB IRC", linewidth=2)
    plt.plot(x, evb, label="EVB barrier-calibrated", linewidth=2)
    plt.xlabel("IRC frame")
    plt.ylabel("Relative energy (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "irc_qm_vs_evb_barrier_calibrated_kcal_mol.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, gap, linewidth=2)
    plt.xlabel("IRC frame")
    plt.ylabel("EVB gap (kcal/mol)")
    plt.tight_layout()
    plt.savefig(OUT / "irc_gap_barrier_calibrated_kcal_mol.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, w1, label="state 1 weight", linewidth=2)
    plt.plot(x, w2, label="state 2 weight", linewidth=2)
    plt.xlabel("IRC frame")
    plt.ylabel("EVB weight")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "irc_weights_barrier_calibrated.png", dpi=200)
    plt.close()


def _write_reports(metrics: dict) -> None:
    (OUT / "irc_evb_barrier_calibrated_report.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# HG3.17 Barrier-Calibrated EVB",
        "",
        f"- delta_alpha: `{metrics['delta_alpha_kcal_mol']:.6f}` kcal/mol",
        f"- H12: `{metrics['h12_kcal_mol']:.6f}` kcal/mol",
        f"- g-xTB IRC barrier: `{metrics['qm_barrier_kcal_mol']:.6f}` kcal/mol",
        f"- EVB barrier: `{metrics['evb_barrier_kcal_mol']:.6f}` kcal/mol",
        f"- g-xTB reaction energy: `{metrics['qm_reaction_energy_kcal_mol']:.6f}` kcal/mol",
        f"- EVB reaction energy: `{metrics['evb_reaction_energy_kcal_mol']:.6f}` kcal/mol",
        f"- RMSE across finite IRC points: `{metrics['rmse_kcal_mol']:.6f}` kcal/mol",
        f"- Max absolute residual: `{metrics['max_abs_residual_kcal_mol']:.6f}` kcal/mol at frame `{metrics['worst_residual_frame']}`",
        "",
        "## Warning",
        "",
        metrics["warning"],
    ]
    (OUT / "irc_evb_barrier_calibrated_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_production_configs(metrics: dict) -> None:
    config_dir = RUN_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for replica_index in range(1, N_REPLICATES + 1):
        (config_dir / f"rep{replica_index:02d}.yaml").write_text(_render_config(replica_index), encoding="utf-8")
    EXAMPLE_CONFIG.write_text(_render_config(1, output_dir="outputs/hg317_irc_mapping_barrier_calibrated"), encoding="utf-8")
    manifest = {
        **metrics,
        "description": "HG3.17 IRC barrier-calibrated EVB mapped-Hamiltonian PMF production setup.",
        "requires_platform": "CUDA",
        "n_replicates": N_REPLICATES,
        "n_windows": N_WINDOWS,
        "lambda_values": _lambda_values(),
        "timestep_fs": TIMESTEP_FS,
        "equilibration_ps_per_window": EQUILIBRATION_PS,
        "production_ps_per_window": PRODUCTION_PS,
        "total_production_ns": N_REPLICATES * N_WINDOWS * PRODUCTION_PS * 1.0e-3,
        "gpu_launcher": str((RUN_ROOT / "run_all_cuda_2gpu.sh").relative_to(ROOT)),
        "example_config": str(EXAMPLE_CONFIG.relative_to(ROOT)),
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    launcher = RUN_ROOT / "run_all_cuda_2gpu.sh"
    launcher.write_text(_render_launcher(), encoding="utf-8")
    launcher.chmod(0o755)


def _lambda_values() -> list[float]:
    return [round(index / (N_WINDOWS - 1), 6) for index in range(N_WINDOWS)]


def _steps(ps: float) -> int:
    return int(round(ps * 1000.0 / TIMESTEP_FS))


def _render_config(replica_index: int, output_dir: str | None = None) -> str:
    seed = SEED_BASE + replica_index
    lambdas = ", ".join(f"{value:.6g}" for value in _lambda_values())
    substrate_atoms = ", ".join(str(index) for index in SUBSTRATE_ATOMS)
    output = output_dir or f"outputs/hg317_irc_mapping_barrier_calibrated_replicates/rep{replica_index:02d}"
    return f"""project:
  name: hg317-irc-mapping-barrier-calibrated-rep{replica_index:02d}
  output_dir: {output}
reaction:
  metadata:
    name: hg317_irc_mapping_barrier_calibrated_rep{replica_index:02d}
    phase: enzyme
    temperature_k: 300.0
    pressure_bar: 1.0
    notes: >
      HG3.17 matched AMBER-state mapped-Hamiltonian PMF. Constant-H12 EVB is calibrated to the g-xTB IRC barrier.
  atoms:
    donor: {DONOR_ATOM}
    proton: {PROTON_ATOM}
    acceptor: {ACCEPTOR_ATOM}
  substrate_atoms: [{substrate_atoms}]
states:
  state1:
    topology: prep/hg317_full_irc/state1_reactant_matched16.prmtop
    coordinates: prep/hg317_full_irc/state1_reactant.inpcrd
    format: amber
  state2:
    topology: prep/hg317_full_irc/state2_product_matched16.prmtop
    coordinates: prep/hg317_full_irc/state1_reactant.inpcrd
    format: amber
evb:
  coupling_model:
    model: constant
    parameters:
      delta_alpha_kj_mol: {DELTA_ALPHA_KJ_MOL:.12g}
      h12_kj_mol: {H12_KJ_MOL:.12g}
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
observables:
  gap:
    shifted: true
  reaction_coordinates:
    - name: proton_transfer_rc
      kind: difference_of_distances
      atom1: {DONOR_ATOM}
      atom2: {PROTON_ATOM}
      atom3: {ACCEPTOR_ATOM}
      event_threshold_nm: 0.0
  distances:
    - name: donor_h
      atom1: {DONOR_ATOM}
      atom2: {PROTON_ATOM}
    - name: h_acceptor
      atom1: {PROTON_ATOM}
      atom2: {ACCEPTOR_ATOM}
analysis:
  histogram:
    bin_min_kj_mol: -100000.0
    bin_max_kj_mol: 300000.0
    n_bins: 500
  pmf:
    temperature_k: 300.0
    zero_mode: reactant_min
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

mkdir -p runs/hg317_irc_mapping_barrier_calibrated/logs

run_replica() {
  local cfg="$1"
  local gpu="$2"
  local rep
  rep="$(basename "$cfg" .yaml)"
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[$(date -Is)] GPU $gpu validate $rep"
  python -m evb.cli validate --config "$cfg" > "runs/hg317_irc_mapping_barrier_calibrated/logs/${rep}_validate.log" 2>&1
  echo "[$(date -Is)] GPU $gpu sample-series $rep"
  python -m evb.cli sample-series --config "$cfg" > "runs/hg317_irc_mapping_barrier_calibrated/logs/${rep}_sample.log" 2>&1
  echo "[$(date -Is)] GPU $gpu analyze $rep"
  python -m evb.cli analyze --config "$cfg" > "runs/hg317_irc_mapping_barrier_calibrated/logs/${rep}_analyze.log" 2>&1
}

index=0
for cfg in runs/hg317_irc_mapping_barrier_calibrated/configs/rep*.yaml; do
  gpu=$(( index % 2 ))
  run_replica "$cfg" "$gpu" &
  index=$(( index + 1 ))
  if (( index % 2 == 0 )); then
    wait
  fi
done
wait
"""


if __name__ == "__main__":
    main()
