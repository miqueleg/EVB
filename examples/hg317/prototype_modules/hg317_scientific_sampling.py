
from __future__ import annotations

import csv
import json
import math
import shutil
import stat
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import EVBConfig, load_config
from .evb import EVBHamiltonian, EVBParameters
from .hg317_gxtb_calibration import collect_hg317_reaction_frames, load_gxtb_reference_profile
from .hg317_qregion import _builder, _read_yaml, _set_output_dir, _write_yaml
from .hg317_reproduction import SELECTED_CANDIDATE, prepare_selected_candidate_config
from .hg317_stability import _append_observable_row, _substrate_com, _write_observables
from .io import write_json
from .native_bias import NativeGapBiasTable1D, NativeWellTemperedGapMetadynamics1D
from .openmm_backend import load_positions_file, write_pdb
from .q_region import QRegionSystemBuilder, q_region_spec_from_config, q_region_to_evb_openmm_system
from .simulation import create_integrator, ensure_output_dir

KJ_TO_KCAL = 1.0 / 4.184

CANDIDATES_TO_AUDIT = [
    "local_pme_q_atoms_cutoff_0.8",
    "local_pme_q_atoms_cutoff_1.2",
    "local_pme_all_atoms_cutoff_1.2",
    "local_pme_q_plus_shell_cutoff_2.0",
    "shared_nonbonded_state1",
]


def prepare_hg317_qregion_scientific_sampling(
    config: EVBConfig,
    config_path: str | Path,
    reference_path: str | Path,
    output: str | Path,
    *,
    platform: str | None = None,
    mode: str = "quick",
    screen_steps: int = 1000,
    run_short_screen: bool = True,
    max_screen_candidates: int = 2,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    reference = load_gxtb_reference_profile(reference_path, "relative_to_RC")
    frames, frame_report = collect_hg317_reaction_frames(config, reference)
    if not {"RC", "TS", "PROD"}.issubset(frames):
        missing = sorted({"RC", "TS", "PROD"}.difference(frames))
        raise ValueError(f"Scientific Q-region sampling requires full-system RC/TS/PROD seed coordinates; missing {missing}.")

    selected = prepare_selected_candidate_config(config_path, output_dir)
    candidate_paths = _candidate_paths(selected)
    audit = []
    for name, path in candidate_paths.items():
        if not path.exists():
            continue
        audit.append(_audit_candidate_frames(name, path, frames, platform or config.simulation.platform))
    if not audit:
        raise ValueError("No fitted Q-region candidate configs were available for scientific sampling setup.")

    selected_audit = next((row for row in audit if row["candidate"] == SELECTED_CANDIDATE), audit[0])
    centers = _ts_focused_centers(selected_audit["frame_gaps_kj_mol"], _window_count(mode))
    config_summary = _write_scientific_configs(selected, output_dir, centers, mode)
    scripts = _write_run_scripts(output_dir)

    short = None
    if run_short_screen:
        screen_candidates = [row for row in audit if row["candidate"] in {SELECTED_CANDIDATE, "local_pme_all_atoms_cutoff_1.2", "shared_nonbonded_state1"}][:max_screen_candidates]
        short = run_short_parameter_screen(screen_candidates, frames, output_dir / "short_screen", platform or config.simulation.platform, screen_steps)

    report = {
        "status": "prepared",
        "scientific_interpretation": "This setup uses full-system relaxed IRC RC/TS/PROD seeds and real Q-region gap umbrella/native table MetaD forces. Short screens are diagnostics only, not PMFs.",
        "reference_profile": reference.target_kj_mol,
        "frame_discovery": frame_report,
        "candidate_frame_audit": audit,
        "selected_candidate": selected_audit,
        "window_plan": {"n_windows": len(centers), "centers_kj_mol": centers, "centers_kcal_mol": [c * KJ_TO_KCAL for c in centers]},
        "configs": config_summary,
        "run_scripts": scripts,
        "short_screen": short,
        "failure_criteria": {
            "require_positive_and_negative_gap_sampling": True,
            "require_both_basins_for_barrier": True,
            "barrier_warning_kcal": 2.0,
            "barrier_fail_kcal": 3.0,
            "old_quick_histogram_proxy_is_not_a_valid_pmf": True,
        },
    }
    write_json(output_dir / "scientific_sampling_setup.json", report)
    _write_report(output_dir / "scientific_sampling_setup.md", report)
    return report


def _candidate_paths(selected: Path) -> dict[str, Path]:
    paths = {SELECTED_CANDIDATE: Path(selected)}
    root = Path("outputs/hg317_qregion_gxtb_calibrated/fitted")
    for name in CANDIDATES_TO_AUDIT:
        path = root / name / "fitted_config.yaml"
        if path.exists():
            paths[name] = path
    return paths


def _audit_candidate_frames(candidate: str, config_path: Path, frames: dict[str, np.ndarray], platform: str | None) -> dict[str, Any]:
    config = load_config(config_path)
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    params = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, params.delta_alpha, params.h12)
    rows = []
    for label in ["RC", "TS", "PROD"]:
        e1, e2, evb, w1, w2, gap = _evaluate_q_region(q_system, frames[label], params, platform)
        rows.append({
            "label": label,
            "E1_kj_mol": e1,
            "E2_kj_mol": e2,
            "Eevb_kj_mol": evb,
            "w1": w1,
            "w2": w2,
            "shifted_gap_kj_mol": gap,
            "shifted_gap_kcal_mol": gap * KJ_TO_KCAL,
        })
    return {
        "candidate": candidate,
        "config": str(config_path),
        "exactness_status": q_system.q_region_report.get("exactness_status"),
        "pme_approximation": q_system.q_region_report.get("pme_approximation"),
        "duplicated_full_nonbonded": q_system.q_region_report.get("duplicated_full_nonbonded"),
        "nonbonded_model_changed": q_system.q_region_report.get("nonbonded_model_changed"),
        "frame_rows": rows,
        "frame_gaps_kj_mol": {row["label"]: row["shifted_gap_kj_mol"] for row in rows},
    }


def _evaluate_q_region(q_system: Any, positions_nm: np.ndarray, params: EVBParameters, platform: str | None) -> tuple[float, float, float, float, float, float]:
    import openmm
    from openmm import unit

    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform_obj = openmm.Platform.getPlatformByName(platform) if platform else None
    context = openmm.Context(q_system.system, integrator, platform_obj) if platform_obj else openmm.Context(q_system.system, integrator)
    if q_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in q_system.box_vectors_nm])
    context.setPositions(positions_nm * unit.nanometer)
    vals = [float(v) for v in q_system.evb_force.getCollectiveVariableValues(context)]
    common = float(context.getState(getEnergy=True, groups={30}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    e1 = common + vals[0]
    e2 = common + vals[1]
    evb_res, w1, w2 = EVBHamiltonian(params).lower_eigenvalue(vals[0], vals[1])
    return e1, e2, common + evb_res, float(w1), float(w2), vals[0] - vals[1] - params.delta_alpha


def _window_count(mode: str) -> int:
    return {"quick": 17, "pilot": 41, "production": 81}.get(mode, 17)


def _ts_focused_centers(frame_gaps: dict[str, float], n_windows: int) -> list[float]:
    rc = float(frame_gaps["RC"])
    ts = float(frame_gaps["TS"])
    prod = float(frame_gaps["PROD"])
    low = min(rc, ts, prod)
    high = max(rc, ts, prod)
    anchors = np.asarray([rc, min(0.0, ts), 0.0, ts, max(0.0, ts), prod], dtype=float)
    anchors = np.unique(np.clip(anchors, low, high))
    dense_low = min(0.0, ts) - 750.0
    dense_high = max(0.0, ts) + 750.0
    n_dense = max(7, n_windows // 3)
    dense = np.linspace(max(low, dense_low), min(high, dense_high), n_dense)
    remaining = max(0, n_windows - len(np.unique(np.concatenate([anchors, dense]))))
    broad = np.linspace(low, high, remaining + 2)[1:-1] if remaining else np.asarray([], dtype=float)
    centers = np.unique(np.concatenate([anchors, dense, broad]))
    centers = centers[(centers >= low) & (centers <= high)]
    if len(centers) > n_windows:
        # Keep TS/mixing anchors and downsample the broad remainder.
        mandatory = set(float(x) for x in anchors)
        optional = [float(x) for x in centers if float(x) not in mandatory]
        keep_optional = np.linspace(0, len(optional) - 1, max(0, n_windows - len(mandatory))).round().astype(int) if optional else []
        centers = np.asarray(sorted(mandatory | {optional[i] for i in keep_optional}), dtype=float)
    return _dedupe_centers([float(x) for x in np.sort(centers)], [float(x) for x in anchors], min_spacing_kj=50.0)


def _dedupe_centers(centers: list[float], mandatory: list[float], *, min_spacing_kj: float) -> list[float]:
    mandatory_values = {round(float(x), 8) for x in mandatory}
    out: list[float] = []
    for center in sorted(float(x) for x in centers):
        is_mandatory = round(center, 8) in mandatory_values
        if not out:
            out.append(center)
            continue
        if center - out[-1] >= min_spacing_kj:
            out.append(center)
            continue
        if is_mandatory and round(out[-1], 8) not in mandatory_values:
            out[-1] = center
    return out


def _write_scientific_configs(selected_config: Path, output_dir: Path, centers: list[float], mode: str) -> dict[str, Any]:
    configs = ensure_output_dir(output_dir / "configs")
    payload = _read_yaml(selected_config)
    _set_output_dir(payload, str(output_dir / "umbrella"))
    payload.setdefault("sampling", {})["mode"] = "q_region_gap_umbrella"
    payload["sampling"]["q_region_gap_umbrella"] = {
        "center_source": "gxtb_calibrated_irc_gaps_ts_focused",
        "force_constant_kj_mol2_screened_values": [0.0005, 0.0015, 0.003],
        "production_force_constant_kj_mol2_recommended_start": 0.0015,
        "centers_kj_mol": centers,
        "centers_kcal_mol": [c * KJ_TO_KCAL for c in centers],
        "production_steps_by_mode": {"quick": 2000, "pilot": 50000, "production": 250000},
    }
    umbrella_path = configs / "qregion_scientific_gap_umbrella.yaml"
    _write_yaml(umbrella_path, payload)
    with (configs / "qregion_scientific_gap_umbrella_windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "center_kj_mol", "center_kcal_mol", "recommended_k_gap_kj_mol2"])
        for i, center in enumerate(centers):
            writer.writerow([f"u{i:03d}", center, center * KJ_TO_KCAL, 0.0015])

    metad = _read_yaml(selected_config)
    _set_output_dir(metad, str(output_dir / "metad"))
    native = metad.setdefault("sampling", {}).setdefault("native_gap_bias", {})
    metad["sampling"]["mode"] = "gap_table_metadynamics"
    native.update({
        "method": "well_tempered_metadynamics",
        "cv": "gap",
        # Keep the broad original full-state gap grid until pilot data prove a narrower
        # Q-region range is safe. Short TS-seeded screens can move far into the
        # product-side gap in <1 ps.
        "min_value": -25000.0,
        "max_value": 25000.0,
        "grid_width": 2501,
        "bias_width": 750.0,
        "height_kj_mol": 0.2,
        "bias_factor": 15.0,
        "frequency": 1000,
        "save_frequency": 5000,
        "bias_dir": "native_bias",
        "restart": True,
    })
    metad_path = configs / "qregion_scientific_gap_table_metad.yaml"
    _write_yaml(metad_path, metad)
    return {"umbrella_config": str(umbrella_path), "umbrella_windows_csv": str(configs / "qregion_scientific_gap_umbrella_windows.csv"), "metad_config": str(metad_path)}


def run_short_parameter_screen(candidate_rows: list[dict[str, Any]], frames: dict[str, np.ndarray], output: Path, platform: str | None, steps: int) -> dict[str, Any]:
    output = ensure_output_dir(output)
    umbrella_rows = []
    metad_rows = []
    for candidate in candidate_rows:
        config_path = Path(candidate["config"])
        gaps = candidate["frame_gaps_kj_mol"]
        screen_centers = [gaps["RC"], 0.0, gaps["TS"]]
        for k_gap in [0.0005, 0.0015, 0.003]:
            for center in screen_centers:
                seed_label = _nearest_seed(center, gaps)
                row = _run_short_umbrella(config_path, frames[seed_label], output / "umbrella" / candidate["candidate"] / f"k_{k_gap:g}" / f"center_{_safe(center)}", platform, steps, center, k_gap, seed_label)
                row.update({"candidate": candidate["candidate"], "k_gap": k_gap, "center_kj_mol": center, "seed_label": seed_label})
                umbrella_rows.append(row)
        if candidate["candidate"] == SELECTED_CANDIDATE:
            for seed_label in ["RC", "TS"]:
                for width, height in [(1000.0, 0.1), (750.0, 0.2), (500.0, 0.5)]:
                    row = _run_short_metad(config_path, frames[seed_label], output / "metad" / seed_label / f"width_{int(width)}_height_{height:g}", platform, steps, width, height)
                    row.update({"candidate": candidate["candidate"], "bias_width_kj_mol": width, "height_kj_mol": height, "seed_label": seed_label})
                    metad_rows.append(row)
    _write_rows(output / "umbrella_screen.csv", umbrella_rows)
    _write_rows(output / "metad_screen.csv", metad_rows)
    summary = {"umbrella": umbrella_rows, "metad": metad_rows, "recommendations": _screen_recommendations(umbrella_rows, metad_rows)}
    write_json(output / "short_screen_summary.json", summary)
    return summary


def _run_short_umbrella(config_path: Path, positions_nm: np.ndarray, output: Path, platform: str | None, steps: int, center: float, k_gap: float, seed_label: str) -> dict[str, Any]:
    config = load_config(config_path)
    q_system, evb_system, params = _build_qregion_context_system(config, gap_center=center, k_gap=k_gap)
    return _run_context_md(config, evb_system, params, positions_nm, output, platform, steps, table=None, metad=None, mode="q_region_gap_umbrella_screen")


def _run_short_metad(config_path: Path, positions_nm: np.ndarray, output: Path, platform: str | None, steps: int, width: float, height: float) -> dict[str, Any]:
    config = load_config(config_path)
    table = NativeGapBiasTable1D(-25000.0, 25000.0, 2501)
    metad = NativeWellTemperedGapMetadynamics1D(table, bias_width=width, height_kj_mol=height, bias_factor=15.0, temperature_k=config.simulation.temperature_k, frequency=250, save_frequency=None, bias_dir=ensure_output_dir(output / "native_bias"), restart=False)
    _q_system, evb_system, params = _build_qregion_context_system(config, table=table)
    return _run_context_md(config, evb_system, params, positions_nm, output, platform, steps, table=table, metad=metad, mode="q_region_table_metad_screen")


def _build_qregion_context_system(config: EVBConfig, *, gap_center: float | None = None, k_gap: float | None = None, table: NativeGapBiasTable1D | None = None):
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    params = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, params.delta_alpha, params.h12, native_gap_bias_table=table, gap_umbrella_center=gap_center, gap_umbrella_force_constant=k_gap)
    return q_system, q_region_to_evb_openmm_system(q_system), params


def _run_context_md(config: EVBConfig, evb_system: Any, params: EVBParameters, positions_nm: np.ndarray, output: Path, platform: str | None, steps: int, *, table: NativeGapBiasTable1D | None, metad: NativeWellTemperedGapMetadynamics1D | None, mode: str) -> dict[str, Any]:
    import openmm
    from openmm import unit

    output = ensure_output_dir(output)
    integrator = create_integrator(config.simulation.timestep_fs, config.simulation.temperature_k, config.simulation.friction_per_ps, config.simulation.integrator)
    platform_obj = openmm.Platform.getPlatformByName(platform) if platform else None
    t0 = time.perf_counter()
    context = openmm.Context(evb_system.system, integrator, platform_obj) if platform_obj else openmm.Context(evb_system.system, integrator)
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in evb_system.box_vectors_nm])
    context.setPositions(positions_nm * unit.nanometer)
    context.setVelocitiesToTemperature(config.simulation.temperature_k * unit.kelvin, config.simulation.seed)
    context_time = time.perf_counter() - t0
    rows = []
    initial_com = _substrate_com(positions_nm, config.reaction.substrate_atoms, evb_system.masses_amu)
    report_interval = max(1, min(250, steps // 10 if steps >= 10 else 1))
    _append_observable_row(rows, context, evb_system, config, params, table, step=0, initial_substrate_com_nm=initial_com, masses_amu=evb_system.masses_amu)
    current = 0
    start = time.perf_counter()
    nan = False
    while current < steps:
        advance = min(report_interval, steps - current)
        integrator.step(advance)
        current += advance
        row = _append_observable_row(rows, context, evb_system, config, params, table, step=current, initial_substrate_com_nm=initial_com, masses_amu=evb_system.masses_amu)
        if metad is not None:
            metad.maybe_deposit(current, row["shifted_gap"], evb_system.evb_force, context)
        if any(isinstance(v, (int, float)) and not math.isfinite(float(v)) for v in row.values()):
            nan = True
            break
    wall = time.perf_counter() - start
    final_positions = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    write_pdb(str(output / "final.pdb"), evb_system.topology, final_positions)
    _write_observables(output / "observables.csv", rows)
    gaps = np.asarray([row["shifted_gap"] for row in rows], dtype=float)
    w2 = np.asarray([row["w2"] for row in rows], dtype=float)
    bias = np.asarray([row["table_bias"] for row in rows], dtype=float)
    summary = {
        "mode": mode,
        "completed_steps": int(current),
        "nan_detected": bool(nan),
        "context_creation_time_s": context_time,
        "md_wall_time_s": wall,
        "steps_per_s": current / wall if wall else None,
        "gap_min_kj_mol": float(np.min(gaps)),
        "gap_max_kj_mol": float(np.max(gaps)),
        "gap_mean_kj_mol": float(np.mean(gaps)),
        "gap_std_kj_mol": float(np.std(gaps)),
        "positive_gap_frames": int(np.sum(gaps > 0.0)),
        "near_zero_frames_250kj": int(np.sum(np.abs(gaps) <= 250.0)),
        "w2_min": float(np.min(w2)),
        "w2_max": float(np.max(w2)),
        "bias_max_kj_mol": float(np.max(bias)),
        "gap_left_native_table_grid": bool(table is not None and (float(np.min(gaps)) < table.grid_min or float(np.max(gaps)) > table.grid_max)),
        "duplicated_full_nonbonded": evb_system.energy_decomposition_report.get("duplicated_full_nonbonded") if evb_system.energy_decomposition_report else None,
        "pme_approximation": evb_system.energy_decomposition_report.get("q_region_report", {}).get("pme_approximation") if evb_system.energy_decomposition_report else None,
    }
    write_json(output / "summary.json", summary)
    return summary


def _nearest_seed(center: float, gaps: dict[str, float]) -> str:
    return min(gaps, key=lambda label: abs(float(gaps[label]) - float(center)))


def _safe(value: float) -> str:
    return f"{value:.0f}".replace("-", "m").replace(".", "p")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _screen_recommendations(umbrella_rows: list[dict[str, Any]], metad_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sane_umbrella = [row for row in umbrella_rows if not row.get("nan_detected") and row.get("gap_std_kj_mol", 0.0) > 1.0]
    best_umbrella = sorted(sane_umbrella, key=lambda row: (abs(row.get("gap_mean_kj_mol", 0.0) - row.get("center_kj_mol", 0.0)), -row.get("gap_std_kj_mol", 0.0)))[:5]
    sane_metad = [row for row in metad_rows if not row.get("nan_detected") and not row.get("gap_left_native_table_grid")]
    best_metad = sorted(sane_metad, key=lambda row: (-row.get("near_zero_frames_250kj", 0), -row.get("positive_gap_frames", 0), abs(row.get("gap_mean_kj_mol", 0.0))))[:3]
    return {
        "umbrella": "Use real Q-region umbrella windows seeded from relaxed IRC frames. Start production from the TS-focused window plan; adjust k after inspecting center tracking and adjacent overlap.",
        "umbrella_best_short_rows": best_umbrella,
        "metad": "Use native table MetaD from RC and TS seeds; reject settings that do not reach gap=0/product-side in pilot runs.",
        "metad_best_short_rows": best_metad,
    }


def _write_run_scripts(output_dir: Path) -> dict[str, str]:
    run_dir = ensure_output_dir(output_dir / "run_commands")
    scripts = {
        "run_short_screen.sh": f"""#!/usr/bin/env bash\nset -euo pipefail\n.venv/bin/evb hg317-qregion-scientific-sampling \\\n  --config examples/hg317_evb_gap_metad.yaml \\\n  --reference examples/hg317_gxtb_reference_profile.yaml \\\n  --output {output_dir} \\\n  --platform CUDA \\\n  --mode quick \\\n  --screen-steps 2000\n""",
        "prepare_production_inputs.sh": f"""#!/usr/bin/env bash\nset -euo pipefail\n.venv/bin/evb hg317-qregion-scientific-sampling \\\n  --config examples/hg317_evb_gap_metad.yaml \\\n  --reference examples/hg317_gxtb_reference_profile.yaml \\\n  --output {output_dir} \\\n  --platform CUDA \\\n  --mode production \\\n  --no-run-short-screen\n""",
    }
    written = {}
    for name, text in scripts.items():
        path = run_dir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written[name] = str(path)
    return written


def _write_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# HG3.17 Q-region Scientific Sampling Setup\n\n")
        handle.write("This report replaces the failed quick histogram proxy with real Q-region umbrella and native table MetaD setup.\n\n")
        handle.write("## Selected Candidate\n\n")
        handle.write(f"- candidate: {report['selected_candidate']['candidate']}\n")
        handle.write(f"- exactness_status: {report['selected_candidate']['exactness_status']}\n")
        handle.write(f"- pme_approximation: {report['selected_candidate']['pme_approximation']}\n")
        handle.write("\n## IRC Frame Gaps\n\n")
        handle.write("| candidate | RC gap kcal/mol | TS gap kcal/mol | PROD gap kcal/mol |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for row in report["candidate_frame_audit"]:
            gaps = row["frame_gaps_kj_mol"]
            handle.write(f"| {row['candidate']} | {gaps['RC']*KJ_TO_KCAL:.3f} | {gaps['TS']*KJ_TO_KCAL:.3f} | {gaps['PROD']*KJ_TO_KCAL:.3f} |\n")
        handle.write("\n## Window Plan\n\n")
        handle.write(f"- n_windows: {report['window_plan']['n_windows']}\n")
        handle.write(f"- first_center_kcal_mol: {report['window_plan']['centers_kcal_mol'][0]:.3f}\n")
        handle.write(f"- last_center_kcal_mol: {report['window_plan']['centers_kcal_mol'][-1]:.3f}\n")
        handle.write("\n## Scientific Criteria\n\n")
        handle.write("A production PMF is valid only if both gap basins and the gap=0/TS region are sampled with overlap. Static RC/TS/PROD fitting alone is not validation.\n")
