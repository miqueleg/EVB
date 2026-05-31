from __future__ import annotations

from pathlib import Path


N_REPLICATES = 30
WINDOW_CENTERS_KJ_MOL = [float(-46000 + 600 * index) for index in range(51)]
SEED_BASE = 6100


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "runs" / "weekend_kemp_solvent"
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for replica_index in range(1, N_REPLICATES + 1):
        config_path = config_dir / f"rep{replica_index:02d}.yaml"
        config_path.write_text(_render_config(replica_index), encoding="utf-8")

    manifest = {
        "n_replicates": N_REPLICATES,
        "n_windows": len(WINDOW_CENTERS_KJ_MOL),
        "window_centers_kj_mol": WINDOW_CENTERS_KJ_MOL,
        "production_ps_per_window": 200.0,
        "evb_time_ns_per_replica": 10.2,
        "evb_time_ns_total": N_REPLICATES * 10.2,
        "note": (
            "This matches 30 replicas x 51 EVB windows x 200 ps/window for a single starting conformation. "
            "The 612 ns/system figure in the cited paper includes both 'in' and 'out' substrate conformations."
        ),
    }
    import json

    with (run_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote {N_REPLICATES} replica configs to {config_dir}")
    print(f"Manifest: {run_root / 'manifest.json'}")


def _render_config(replica_index: int) -> str:
    seed = SEED_BASE + replica_index
    output_dir = f"outputs/kemp_solvent_weekend/rep{replica_index:02d}"
    centers = ", ".join(f"{center:.1f}" for center in WINDOW_CENTERS_KJ_MOL)
    return f"""project:
  name: kemp-solvent-weekend-rep{replica_index:02d}
  output_dir: {output_dir}
reaction:
  metadata:
    name: kemp_solvent_weekend_rep{replica_index:02d}
    phase: solution
    temperature_k: 300.0
    pressure_bar: 1.0
    notes: >
      Long CUDA EVB umbrella replicate prepared to match 30 replicas x 51 EVB windows x 200 ps/window
      for the current shared-box Kemp solvent system.
  atoms:
    donor: 17
    proton: 22
    acceptor: 19
states:
  state1:
    topology: outputs/prepared_systems/kemp_solvent_boxsynced/RC.prmtop
    coordinates: outputs/prepared_systems/kemp_solvent_boxsynced/RC.pdb
    format: amber
  state2:
    topology: outputs/prepared_systems/kemp_solvent_boxsynced/PROD.prmtop
    coordinates: outputs/prepared_systems/kemp_solvent_boxsynced/PROD.pdb
    format: amber
evb:
  coupling_model:
    model: constant
    parameters:
      delta_alpha_kj_mol: 0.0
      h12_kj_mol: 20.0
sampling:
  mode: gap_umbrella
  integrator:
    name: LangevinMiddle
    timestep_fs: 0.1
    friction_per_ps: 1.0
    seed: {seed}
  md:
    equilibration_steps: 20000
    production_steps: 2000000
    report_stride: 10000
    save_stride: null
    platform: CUDA
    temperature_k: 300.0
    nonbonded_method: PME
    constraints: None
    minimize_steps: 1000
    minimize_tolerance: 10.0
  windows:
    gap_umbrella:
      centers_kj_mol: [{centers}]
      force_constant_kj_mol2: 1.0e-7
observables:
  gap:
    shifted: true
  distances:
    - name: donor_h
      atom1: 17
      atom2: 22
    - name: h_acceptor
      atom1: 22
      atom2: 19
analysis:
  histogram:
    bin_min_kj_mol: -60000.0
    bin_max_kj_mol: 20000.0
    n_bins: 320
  pmf:
    temperature_k: 300.0
    zero_mode: reactant_min
  barrier:
    reactant_region: [-42000.0, -28000.0]
    product_region: [-24000.0, -14000.0]
"""


if __name__ == "__main__":
    main()
