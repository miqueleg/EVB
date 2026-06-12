from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EVBConfig, load_config
from .evb import EVBHamiltonian, EVBParameters
from .hg317_qregion import (
    _candidate_name,
    _q_region_diabatic_energies,
    _read_yaml,
    _set_output_dir,
    _set_parameters,
    _write_yaml,
    make_hg317_qregion_candidates,
    run_hg317_qregion_smoke,
)
from .io import write_json
from .openmm_backend import AmberSystemLoader, EVBSystemBuilder, load_positions_file
from .q_region import QRegionSystemBuilder, q_region_spec_from_config
from .simulation import ensure_output_dir

KCAL_TO_KJ = 4.184


@dataclass(slots=True)
class GxtbReferenceProfile:
    system: str
    method: str
    profile_name: str
    target_kcal_mol: dict[str, float]
    target_kj_mol: dict[str, float]
    states: dict[str, dict[str, float]]


@dataclass(slots=True)
class GxtbFitResult:
    delta_alpha_kj_mol: float
    h12_kj_mol: float
    rms_residual_kj_mol: float
    max_residual_kj_mol: float
    model_relative_kj_mol: list[float]
    residual_kj_mol: list[float]
    weights1: list[float]
    weights2: list[float]
    shifted_gap_kj_mol: list[float]


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required for HG3.17 g-xTB calibration workflows.") from exc
    return yaml


def load_gxtb_reference_profile(path: str | Path, profile_name: str | None = None) -> GxtbReferenceProfile:
    yaml = _yaml_module()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    selected = profile_name or payload.get("recommended_calibration_profile", "relative_to_RC")
    profiles = payload.get("reaction_profiles", {})
    if selected not in profiles:
        raise ValueError(f"Reference profile {selected!r} not found in {path}.")
    profile = profiles[selected]
    units = profile.get("units", "kcal/mol").lower()
    target_kcal = {key: float(value) for key, value in profile.items() if key != "units"}
    if units in {"kj/mol", "kjmol", "kilojoule/mol"}:
        target_kj = dict(target_kcal)
        target_kcal = {key: value / KCAL_TO_KJ for key, value in target_kj.items()}
    elif units in {"kcal/mol", "kcalmol"}:
        target_kj = {key: value * KCAL_TO_KJ for key, value in target_kcal.items()}
    else:
        raise ValueError(f"Unsupported reference profile units: {profile.get('units')}")
    return GxtbReferenceProfile(
        system=str(payload.get("system", "HG3.17")),
        method=str(payload.get("method", "g-xtb")),
        profile_name=selected,
        target_kcal_mol=target_kcal,
        target_kj_mol=target_kj,
        states=payload.get("states", {}),
    )


def collect_hg317_reaction_frames(config: EVBConfig, reference_profile: GxtbReferenceProfile) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Find full-system coordinates for the requested g-xTB profile labels."""
    wanted = [label for label in reference_profile.target_kj_mol if label in {"Bound", "RC", "TS", "PROD"}]
    if not wanted:
        wanted = ["RC", "TS", "PROD"]
    report: list[dict[str, Any]] = []
    frames: dict[str, np.ndarray] = {}

    setup_report = Path("prep/hg317_full_irc/evb_ready/setup_from_irc/analysis/evb_reference_fit_from_irc.json")
    seed_csv = Path("prep/hg317_full_irc/evb_ready/setup_from_irc/analysis/irc_relaxed_seeds/irc_seed_relaxation.csv")
    selected: dict[str, int] = {}
    if setup_report.exists():
        with setup_report.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        selected_frames = payload.get("selected_frames", {})
        for label, key in (("RC", "rc"), ("TS", "ts"), ("PROD", "product")):
            if key in selected_frames:
                selected[label] = int(selected_frames[key]["canonical_frame"])

    seed_paths: dict[int, str] = {}
    if seed_csv.exists():
        with seed_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                seed_paths[int(row["canonical_frame"])] = row["pdb"]

    for label in wanted:
        source_path: str | None = None
        notes: list[str] = []
        if label in selected and selected[label] in seed_paths:
            source_path = seed_paths[selected[label]]
            notes.append(f"using setup-from-irc relaxed seed canonical frame {selected[label]}")
        elif label == "RC":
            source_path = config.state1.coordinates
            notes.append("fallback to state1 coordinates")
        elif label == "PROD":
            source_path = config.state2.coordinates
            notes.append("fallback to state2 coordinates")
        elif Path("examples/HD3.17_IRC.xyz").exists():
            notes.append("examples/HD3.17_IRC.xyz is cluster-only and was not used as full-system coordinates")

        record = {
            "frame_id": label,
            "label": label,
            "source_path": source_path,
            "coordinate_format": None if source_path is None else Path(source_path).suffix.lstrip("."),
            "full_system_coordinates_available": bool(source_path and Path(source_path).exists()),
            "can_be_used_for_calibration": False,
            "notes": notes,
        }
        if source_path and Path(source_path).exists():
            frames[label] = load_positions_file(source_path)
            record["can_be_used_for_calibration"] = True
        report.append(record)
    return frames, report


def _evb_profile(e1: np.ndarray, e2: np.ndarray, delta_alpha: float, h12: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ham = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha, h12=h12))
    evb = []
    w1 = []
    w2 = []
    gap = []
    for left, right in zip(e1, e2):
        energy, weight1, weight2 = ham.lower_eigenvalue(float(left), float(right))
        evb.append(energy)
        w1.append(weight1)
        w2.append(weight2)
        gap.append(float(left - right - delta_alpha))
    evb_arr = np.asarray(evb, dtype=float)
    return evb_arr - evb_arr[0], np.asarray(w1), np.asarray(w2), np.asarray(gap)


def _score_grid(e1: np.ndarray, e2: np.ndarray, target: np.ndarray, delta_values: np.ndarray, h12_values: np.ndarray) -> tuple[float, float, float]:
    best = (float("inf"), float(delta_values[0]), float(h12_values[0]))
    for delta in delta_values:
        gap = e1 - e2 - float(delta)
        shifted2 = e2 + float(delta)
        for h12 in h12_values:
            root = np.sqrt(0.25 * gap * gap + float(h12) * float(h12))
            values = 0.5 * (e1 + shifted2) - root
            rel = values - values[0]
            residual = rel - target
            rms = float(np.sqrt(np.mean(residual * residual)))
            if rms < best[0]:
                best = (rms, float(delta), float(h12))
    return best


def fit_gxtb_profile(
    labels: list[str],
    e1_kj_mol: list[float],
    e2_kj_mol: list[float],
    target_kj_mol: list[float],
    *,
    delta_alpha_initial_kj_mol: float,
    h12_initial_kj_mol: float,
    output_dir: str | Path | None = None,
) -> GxtbFitResult:
    e1 = np.asarray(e1_kj_mol, dtype=float)
    e2 = np.asarray(e2_kj_mol, dtype=float)
    target = np.asarray(target_kj_mol, dtype=float)
    target = target - target[0]
    if len(e1) < 3:
        raise ValueError("At least three labeled frames are required for g-xTB EVB fitting.")

    raw_gap = e1 - e2
    raw_center = float(np.median(raw_gap))
    initial = float(delta_alpha_initial_kj_mol)
    span = max(float(np.nanmax(raw_gap) - np.nanmin(raw_gap)), float(np.nanmax(target) - np.nanmin(target)), abs(raw_center - initial), 50000.0)
    delta_lo = min(raw_center, initial) - span
    delta_hi = max(raw_center, initial) + span
    h_hi = max(2.0 * abs(float(h12_initial_kj_mol)), 2000.0)

    coarse_delta = np.linspace(delta_lo, delta_hi, 401)
    coarse_h12 = np.linspace(0.0, h_hi, 201)
    _write_scan(output_dir, "fit_scan_coarse.csv", labels, e1, e2, target, coarse_delta, coarse_h12)
    _rms, best_delta, best_h12 = _score_grid(e1, e2, target, coarse_delta, coarse_h12)

    delta_width = max((delta_hi - delta_lo) / 200.0, 10.0)
    h_width = max(h_hi / 100.0, 2.0)
    refined_delta = np.linspace(best_delta - delta_width, best_delta + delta_width, 401)
    refined_h12 = np.linspace(max(0.0, best_h12 - h_width), best_h12 + h_width, 201)
    _write_scan(output_dir, "fit_scan_refined.csv", labels, e1, e2, target, refined_delta, refined_h12)
    _rms, best_delta, best_h12 = _score_grid(e1, e2, target, refined_delta, refined_h12)

    for _ in range(3):
        delta_width = max(delta_width / 20.0, 1.0e-4)
        h_width = max(h_width / 20.0, 1.0e-4)
        local_delta = np.linspace(best_delta - delta_width, best_delta + delta_width, 201)
        local_h12 = np.linspace(max(0.0, best_h12 - h_width), best_h12 + h_width, 201)
        _rms, best_delta, best_h12 = _score_grid(e1, e2, target, local_delta, local_h12)

    best_delta, best_h12 = _optimize_fit_with_scipy(
        e1,
        e2,
        target,
        [(best_delta, best_h12), (initial, float(h12_initial_kj_mol)), (raw_center, float(h12_initial_kj_mol)), (float(raw_gap[0]), float(h12_initial_kj_mol))],
    )

    profile, w1, w2, shifted_gap = _evb_profile(e1, e2, best_delta, best_h12)
    residual = profile - target
    return GxtbFitResult(
        delta_alpha_kj_mol=float(best_delta),
        h12_kj_mol=float(best_h12),
        rms_residual_kj_mol=float(np.sqrt(np.mean(residual * residual))),
        max_residual_kj_mol=float(np.max(np.abs(residual))),
        model_relative_kj_mol=[float(value) for value in profile],
        residual_kj_mol=[float(value) for value in residual],
        weights1=[float(value) for value in w1],
        weights2=[float(value) for value in w2],
        shifted_gap_kj_mol=[float(value) for value in shifted_gap],
    )


def _fit_objective(e1: np.ndarray, e2: np.ndarray, target: np.ndarray, delta_alpha: float, h12: float) -> float:
    profile, _w1, _w2, _gap = _evb_profile(e1, e2, float(delta_alpha), max(0.0, float(h12)))
    residual = profile - target
    return float(np.sqrt(np.mean(residual * residual)))


def _optimize_fit_with_scipy(e1: np.ndarray, e2: np.ndarray, target: np.ndarray, seeds: list[tuple[float, float]]) -> tuple[float, float]:
    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover - grid result remains valid without scipy
        return seeds[0]

    best = (_fit_objective(e1, e2, target, seeds[0][0], seeds[0][1]), float(seeds[0][0]), max(0.0, float(seeds[0][1])))
    for delta_seed, h12_seed in seeds:
        result = minimize(
            lambda x: _fit_objective(e1, e2, target, float(x[0]), float(x[1])),
            np.asarray([float(delta_seed), max(0.0, float(h12_seed))]),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1.0e-8, "fatol": 1.0e-8},
        )
        delta = float(result.x[0])
        h12 = max(0.0, float(result.x[1]))
        score = _fit_objective(e1, e2, target, delta, h12)
        if score < best[0]:
            best = (score, delta, h12)
    return best[1], best[2]


def _write_scan(output_dir: str | Path | None, filename: str, labels: list[str], e1: np.ndarray, e2: np.ndarray, target: np.ndarray, deltas: np.ndarray, h12s: np.ndarray) -> None:
    if output_dir is None:
        return
    destination = ensure_output_dir(output_dir) / filename
    # Keep scan files compact: store the best H12 for each delta row.
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delta_alpha_kj_mol", "best_h12_kj_mol", "rms_residual_kj_mol"])
        for delta in deltas:
            rms, _delta, h12 = _score_grid(e1, e2, target, np.asarray([delta]), h12s)
            writer.writerow([float(delta), float(h12), float(rms)])


def _builder(config: EVBConfig) -> EVBSystemBuilder:
    return EVBSystemBuilder(AmberSystemLoader(config.simulation.nonbonded_method, config.simulation.constraints))


def _evaluate_candidate_frames(config_path: str | Path, reference: GxtbReferenceProfile, platform: str | None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    config = load_config(config_path)
    if platform:
        config.simulation.platform = platform
    frames, frame_report = collect_hg317_reaction_frames(config, reference)
    labels = [label for label in reference.target_kj_mol if label in frames]
    if len(labels) < 3:
        missing = [label for label in reference.target_kj_mol if label not in frames]
        raise ValueError(f"Missing full-system calibration frames: {missing}")
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    original = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, original.delta_alpha, original.h12)
    rows = []
    for label in labels:
        e1, e2 = _q_region_diabatic_energies(q_system, frames[label], platform_name=platform or "CPU")
        rows.append(
            {
                "label": label,
                "target_kj_mol": float(reference.target_kj_mol[label]),
                "E1_kj_mol": e1,
                "E2_kj_mol": e2,
                "raw_gap_kj_mol": e1 - e2,
            }
        )
    metadata = {
        "labels": labels,
        "original_parameters": asdict(original),
        "q_region_report": q_system.q_region_report,
        "frame_discovery": frame_report,
    }
    return metadata, rows, frames


def calibrate_hg317_qregion_gxtb(
    config: EVBConfig,
    config_path: str | Path,
    reference_path: str | Path,
    candidate_dir: str | Path,
    output: str | Path,
    profile: str = "relative_to_RC",
    platform: str | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    reference = load_gxtb_reference_profile(reference_path, profile)
    config_dir = Path(candidate_dir)
    if not config_dir.exists() or not list(config_dir.glob("*.yaml")):
        generated = make_hg317_qregion_candidates(config, config_path, output_dir / "generated_candidates", include_reaction_atoms=True)
        config_dir = Path(generated["candidate_dir"])
    fitted_dir = ensure_output_dir(output_dir / "fitted")
    records = []
    for candidate_path in sorted(config_dir.glob("*.yaml")):
        name = _candidate_name(candidate_path)
        candidate_output = ensure_output_dir(fitted_dir / name)
        try:
            metadata, energy_rows, _frames = _evaluate_candidate_frames(candidate_path, reference, platform)
            labels = metadata["labels"]
            original = metadata["original_parameters"]
            fit = fit_gxtb_profile(
                labels,
                [row["E1_kj_mol"] for row in energy_rows],
                [row["E2_kj_mol"] for row in energy_rows],
                [row["target_kj_mol"] for row in energy_rows],
                delta_alpha_initial_kj_mol=original["delta_alpha"],
                h12_initial_kj_mol=original["h12"],
                output_dir=candidate_output,
            )
            fitted_config = candidate_output / "fitted_config.yaml"
            write_gxtb_fitted_config(
                candidate_path,
                fitted_config,
                EVBParameters(fit.delta_alpha_kj_mol, fit.h12_kj_mol),
                reference,
                fit,
                candidate_output / "run",
            )
            _write_profile_csv(candidate_output / "profile_fit.csv", labels, energy_rows, fit)
            report = {
                "candidate": name,
                "candidate_config": str(candidate_path),
                "fitted_config": str(fitted_config),
                "reference_profile": asdict(reference),
                "fit_success": True,
                "original_parameters": original,
                "fitted_parameters": {"delta_alpha": fit.delta_alpha_kj_mol, "h12": fit.h12_kj_mol},
                "fit": asdict(fit),
                "energy_rows": energy_rows,
                "q_region_report": metadata["q_region_report"],
                "frame_discovery": metadata["frame_discovery"],
            }
        except Exception as exc:
            report = {
                "candidate": name,
                "candidate_config": str(candidate_path),
                "fit_success": False,
                "error": str(exc),
            }
        write_json(candidate_output / "fit_report.json", report)
        if report.get("fit_success"):
            write_json(candidate_output / "fitted_parameters.json", report["fitted_parameters"])
        records.append(report)
    summary = {"reference_profile": asdict(reference), "candidates": [_compact_fit_record(row) for row in records]}
    write_json(output_dir / "calibration_summary.json", summary)
    _write_calibration_summary_csv(output_dir / "calibration_summary.csv", records)
    return summary


def _write_profile_csv(path: Path, labels: list[str], energy_rows: list[dict[str, Any]], fit: GxtbFitResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "target_energy_kj_mol", "model_energy_after_fit_kj_mol", "residual_kj_mol", "E1_kj_mol", "E2_kj_mol", "raw_gap_kj_mol", "shifted_gap_kj_mol", "w1", "w2"])
        for idx, (label, row) in enumerate(zip(labels, energy_rows)):
            writer.writerow([
                label,
                row["target_kj_mol"],
                fit.model_relative_kj_mol[idx],
                fit.residual_kj_mol[idx],
                row["E1_kj_mol"],
                row["E2_kj_mol"],
                row["raw_gap_kj_mol"],
                fit.shifted_gap_kj_mol[idx],
                fit.weights1[idx],
                fit.weights2[idx],
            ])


def write_gxtb_fitted_config(
    candidate_config_path: str | Path,
    output_path: str | Path,
    parameters: EVBParameters,
    reference: GxtbReferenceProfile,
    fit: GxtbFitResult,
    output_dir: str | Path,
) -> None:
    payload = _read_yaml(candidate_config_path)
    _set_parameters(payload, parameters)
    _set_output_dir(payload, str(output_dir))
    payload.setdefault("evb", {}).setdefault("q_region", {})["calibration"] = {
        "source": "gxtb_reference_profile",
        "profile": reference.profile_name,
        "target_kj_mol": reference.target_kj_mol,
        "rms_residual_kj_mol": fit.rms_residual_kj_mol,
        "max_residual_kj_mol": fit.max_residual_kj_mol,
        "status": "gxtb_calibrated",
    }
    _write_yaml(output_path, payload)


def _compact_fit_record(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("fit_success"):
        return {"candidate": row.get("candidate"), "fit_success": False, "error": row.get("error")}
    q_report = row.get("q_region_report", {})
    return {
        "candidate": row["candidate"],
        "fit_success": True,
        "fitted_config": row["fitted_config"],
        "mode": q_report.get("nonbonded_mode"),
        "exactness_status": q_report.get("exactness_status"),
        "pme_approximation": q_report.get("pme_approximation"),
        "nonbonded_model_changed": q_report.get("nonbonded_model_changed"),
        "duplicated_full_nonbonded": q_report.get("duplicated_full_nonbonded"),
        "delta_alpha_old": row["original_parameters"]["delta_alpha"],
        "delta_alpha_new": row["fitted_parameters"]["delta_alpha"],
        "h12_old": row["original_parameters"]["h12"],
        "h12_new": row["fitted_parameters"]["h12"],
        "rms_residual_kj_mol": row["fit"]["rms_residual_kj_mol"],
        "max_residual_kj_mol": row["fit"]["max_residual_kj_mol"],
    }


def _write_calibration_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["candidate", "fit_success", "mode", "delta_alpha_old", "delta_alpha_new", "h12_old", "h12_new", "rms_residual_kj_mol", "max_residual_kj_mol", "duplicated_full_nonbonded", "pme_approximation", "nonbonded_model_changed", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _compact_fit_record(record).get(key) for key in fieldnames})


def _force_sanity(config_path: str | Path, labels: list[str], frames: dict[str, np.ndarray], platform: str | None) -> dict[str, Any]:
    import openmm
    from openmm import unit

    config = load_config(config_path)
    if platform:
        config.simulation.platform = platform
    builder = _builder(config)
    state1, state2 = builder.build_from_state_files(config.state1, config.state2)
    params = EVBParameters(config.evb_parameters.delta_alpha or 0.0, config.evb_parameters.h12 or 0.0)
    q_system = QRegionSystemBuilder(q_region_spec_from_config(config)).build(state1, state2, params.delta_alpha, params.h12)
    from .q_region import q_region_to_evb_openmm_system

    evb_system = q_region_to_evb_openmm_system(q_system)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform_obj = openmm.Platform.getPlatformByName(platform) if platform else None
    context = openmm.Context(evb_system.system, integrator, platform_obj) if platform_obj else openmm.Context(evb_system.system, integrator)
    if evb_system.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*[v * unit.nanometer for v in evb_system.box_vectors_nm])
    frame_reports = []
    for label in labels:
        context.setPositions(frames[label] * unit.nanometer)
        state = context.getState(getForces=True)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        frame_reports.append({"label": label, "force_rms_kj_mol_nm": float(np.sqrt(np.mean(forces * forces))), "force_max_abs_kj_mol_nm": float(np.max(np.abs(forces)))})
    return {
        "frames": frame_reports,
        "max_force_abs_kj_mol_nm": max(row["force_max_abs_kj_mol_nm"] for row in frame_reports),
        "max_force_rms_kj_mol_nm": max(row["force_rms_kj_mol_nm"] for row in frame_reports),
        "q_region_report": q_system.q_region_report,
    }


def validate_hg317_qregion_gxtb(calibrated_dir: str | Path, reference_path: str | Path, output: str | Path, platform: str | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    per_candidate_dir = ensure_output_dir(output_dir / "per_candidate")
    reference = load_gxtb_reference_profile(reference_path, None)
    rows = []
    for fit_report_path in sorted(Path(calibrated_dir).glob("fitted/*/fit_report.json")):
        with fit_report_path.open("r", encoding="utf-8") as handle:
            fit_report = json.load(handle)
        if not fit_report.get("fit_success"):
            rows.append({"candidate": fit_report.get("candidate"), "fit_success": False, "exploratory_valid": False, "production_valid": False, "error": fit_report.get("error")})
            continue
        config_path = fit_report["fitted_config"]
        config = load_config(config_path)
        frames, frame_report = collect_hg317_reaction_frames(config, reference)
        labels = [row["label"] for row in fit_report["energy_rows"] if row["label"] in frames]
        force = _force_sanity(config_path, labels, frames, platform)
        compact = _compact_fit_record(fit_report)
        exploratory = (
            compact["rms_residual_kj_mol"] <= 5.0
            and compact["max_residual_kj_mol"] <= 10.0
            and force["max_force_abs_kj_mol_nm"] <= 5000.0
            and compact["duplicated_full_nonbonded"] is False
            and all(math.isfinite(row["force_max_abs_kj_mol_nm"]) for row in force["frames"])
        )
        production = exploratory and compact["rms_residual_kj_mol"] <= 1.0 and compact["max_residual_kj_mol"] <= 2.0
        row = {
            **compact,
            "force_sanity": force,
            "frame_discovery": frame_report,
            "exploratory_valid": bool(exploratory),
            "production_valid": bool(production),
        }
        rows.append(row)
        write_json(per_candidate_dir / f"{compact['candidate']}_gxtb_validation.json", row)
    selected = select_gxtb_candidate(rows)
    summary = {"reference_profile": asdict(reference), "candidates": rows, "selected_candidate": selected}
    write_json(output_dir / "gxtb_validation_summary.json", summary)
    _write_gxtb_validation_csv(output_dir / "gxtb_validation_summary.csv", rows)
    _write_selected(output_dir, selected)
    return summary


def select_gxtb_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [row for row in rows if row.get("fit_success")]
    exploratory = [row for row in usable if row.get("exploratory_valid")]
    pool = exploratory or usable
    if not pool:
        return None
    return min(
        pool,
        key=lambda row: (
            float(row.get("rms_residual_kj_mol") or 1e99),
            float(row.get("max_residual_kj_mol") or 1e99),
            float((row.get("force_sanity") or {}).get("max_force_abs_kj_mol_nm") or 1e99),
        ),
    )


def _write_gxtb_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["candidate", "exactness_status", "pme_approximation", "nonbonded_model_changed", "delta_alpha_new", "h12_new", "rms_residual_kj_mol", "max_residual_kj_mol", "exploratory_valid", "production_valid", "duplicated_full_nonbonded"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_selected(output_dir: Path, selected: dict[str, Any] | None) -> None:
    if selected is None:
        write_json(output_dir / "selected_candidate.json", {"selected": None})
        (output_dir / "selected_candidate.md").write_text("No fitted candidate was available.\n", encoding="utf-8")
        return
    write_json(output_dir / "selected_candidate.json", selected)
    with (output_dir / "selected_candidate.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# Selected HG3.17 g-xTB Q-Region Candidate\n\n")
        handle.write(f"- candidate: `{selected.get('candidate')}`\n")
        handle.write(f"- exploratory_valid: {selected.get('exploratory_valid')}\n")
        handle.write(f"- production_valid: {selected.get('production_valid')}\n")
        handle.write(f"- RMS residual: {selected.get('rms_residual_kj_mol')} kJ/mol\n")
        handle.write(f"- max residual: {selected.get('max_residual_kj_mol')} kJ/mol\n")
        handle.write(f"- duplicated_full_nonbonded: {selected.get('duplicated_full_nonbonded')}\n")


def smoke_hg317_qregion_gxtb(calibrated_dir: str | Path, output: str | Path, platform: str | None, steps: int = 2000, smoke_policy: str = "exploratory") -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    validation_path = Path(calibrated_dir) / "validation" / "gxtb_validation_summary.json"
    sibling_validation_path = Path(calibrated_dir).parent / "validation" / "gxtb_validation_summary.json"
    if not validation_path.exists() and sibling_validation_path.exists():
        validation_path = sibling_validation_path
    if validation_path.exists():
        with validation_path.open("r", encoding="utf-8") as handle:
            validation = json.load(handle)
    else:
        validation = validate_hg317_qregion_gxtb(calibrated_dir, Path("examples/hg317_gxtb_reference_profile.yaml"), Path(calibrated_dir) / "validation", platform)
    selected = validation.get("selected_candidate")
    if selected is None:
        result = {"ran": False, "reason": "No fitted candidate is available."}
    elif not selected.get("exploratory_valid") and smoke_policy != "force":
        result = {"ran": False, "reason": "Selected candidate is not exploratory_valid; rerun with --smoke-policy force for an explicit forced smoke.", "selected_candidate": selected.get("candidate")}
    else:
        result = run_hg317_qregion_smoke(selected["fitted_config"], output_dir, platform, steps, forced=not selected.get("exploratory_valid"))
        result["candidate"] = selected.get("candidate")
        result["exploratory_valid"] = selected.get("exploratory_valid")
        result["production_valid"] = selected.get("production_valid")
    write_json(output_dir / "smoke_summary.json", result)
    return result


def hg317_qregion_gxtb_workflow(
    config: EVBConfig,
    config_path: str | Path,
    reference_path: str | Path,
    output: str | Path,
    profile: str,
    platform: str | None,
    smoke_steps: int,
    smoke_policy: str,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    candidate_root = output_dir / "candidates"
    generated = make_hg317_qregion_candidates(config, config_path, candidate_root, include_reaction_atoms=True)
    calibration = calibrate_hg317_qregion_gxtb(config, config_path, reference_path, generated["candidate_dir"], output_dir / "calibration", profile, platform)
    validation = validate_hg317_qregion_gxtb(output_dir / "calibration", reference_path, output_dir / "validation", platform)
    smoke = smoke_hg317_qregion_gxtb(output_dir / "calibration", output_dir / "smoke", platform, smoke_steps, smoke_policy)
    summary = {"candidate_generation": generated, "calibration": calibration, "validation": validation, "smoke": smoke}
    write_json(output_dir / "workflow_summary.json", summary)
    return summary
