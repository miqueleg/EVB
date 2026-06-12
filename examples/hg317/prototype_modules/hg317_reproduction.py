from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EVBConfig, load_config
from .evb import EVBParameters
from .hg317_gxtb_calibration import load_gxtb_reference_profile
from .hg317_qregion import _read_yaml, _set_output_dir, _set_parameters, _write_yaml
from .hg317_stability import run_hg317_qregion_stability_check
from .io import write_json
from .simulation import ensure_output_dir

KJ_TO_KCAL = 1.0 / 4.184
SELECTED_CANDIDATE = "local_pme_q_atoms_cutoff_0.8"
SELECTED_DELTA_ALPHA = 405.2455501867761
SELECTED_H12 = 431.6758380801819


@dataclass(slots=True)
class ReproductionSettings:
    mode: str
    replicas: int
    umbrella_steps: int
    metad_steps: int
    n_windows: int


def settings_for_mode(mode: str, replicas: int | None = None, umbrella_steps: int | None = None, metad_steps: int | None = None) -> ReproductionSettings:
    defaults = {
        "quick": (1, 1000, 5000, 7),
        "pilot": (2, 20000, 100000, 21),
        "production": (3, 200000, 1000000, 51),
    }
    if mode not in defaults:
        raise ValueError("mode must be quick, pilot, or production")
    d_replicas, d_umbrella, d_metad, d_windows = defaults[mode]
    return ReproductionSettings(
        mode=mode,
        replicas=int(replicas or d_replicas),
        umbrella_steps=int(umbrella_steps or d_umbrella),
        metad_steps=int(metad_steps or d_metad),
        n_windows=d_windows,
    )


def find_selected_qregion_config() -> Path | None:
    roots = [
        Path("outputs/hg317_qregion_gxtb_calibrated"),
        Path("outputs/hg317_qregion_gxtb_workflow"),
        Path("outputs/hg317_qregion_calibrated"),
    ]
    for root in roots:
        selected = root / "validation" / "selected_candidate.json"
        if selected.exists():
            payload = json.loads(selected.read_text(encoding="utf-8"))
            if payload.get("candidate") == SELECTED_CANDIDATE and payload.get("fitted_config"):
                path = Path(payload["fitted_config"])
                if path.exists():
                    return path
        direct = root / "fitted" / SELECTED_CANDIDATE / "fitted_config.yaml"
        if direct.exists():
            return direct
    return None


def prepare_selected_candidate_config(base_config_path: str | Path, output_dir: str | Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    found = find_selected_qregion_config()
    destination = output_dir / "selected_candidate" / "fitted_config.yaml"
    if found is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, destination)
        return destination
    payload = _read_yaml(base_config_path)
    payload.setdefault("evb", {})["representation"] = "q_region"
    payload["evb"].setdefault("q_region", {})
    payload["evb"]["q_region"].update(
        {
            "enabled": True,
            "selected_candidate": SELECTED_CANDIDATE,
            "baseline_state": "state1",
            "common_force_placement": "outer_system",
            "nonbonded": {
                "mode": "local_pme_approx",
                "pme_policy": "local_direct_space_correction",
                "local_approx_enabled": True,
                "correction_atoms": "q_atoms",
                "correction_cutoff_nm": 0.8,
                "include_q_q": True,
                "include_q_environment": True,
                "include_q_water": True,
                "include_exceptions": True,
            },
        }
    )
    _set_parameters(payload, EVBParameters(SELECTED_DELTA_ALPHA, SELECTED_H12))
    _set_output_dir(payload, str(output_dir / "selected_candidate" / "run"))
    _write_yaml(destination, payload)
    return destination


def generate_reproduction_configs(base_config_path: str | Path, selected_config: str | Path, output: str | Path, settings: ReproductionSettings) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    config_dir = ensure_output_dir(output_dir / "configs")
    selected_payload = _read_yaml(selected_config)
    selected_payload.setdefault("evb", {}).setdefault("q_region", {})["selected_candidate"] = SELECTED_CANDIDATE
    _set_parameters(selected_payload, EVBParameters(SELECTED_DELTA_ALPHA, SELECTED_H12))

    umbrella = json_roundtrip(selected_payload)
    umbrella.setdefault("sampling", {})["mode"] = "q_region_gap_umbrella"
    umbrella["sampling"]["q_region_gap_umbrella"] = {
        "n_windows": settings.n_windows,
        "gap_min_kcal_mol": -2500.0 if settings.mode != "quick" else -500.0,
        "gap_max_kcal_mol": 2500.0 if settings.mode != "quick" else 500.0,
        "force_constant_kcal_mol2": 0.005,
        "production_steps": settings.umbrella_steps,
    }
    _set_output_dir(umbrella, str(output_dir / "umbrella"))
    umbrella_path = config_dir / "qregion_gap_umbrella.yaml"
    _write_yaml(umbrella_path, umbrella)

    metad = json_roundtrip(selected_payload)
    metad.setdefault("sampling", {})["mode"] = "gap_table_metadynamics"
    native = metad["sampling"].setdefault("native_gap_bias", {})
    native.setdefault("method", "well_tempered_metadynamics")
    native.setdefault("cv", "gap")
    native["min_value"] = -5000.0 * 4.184
    native["max_value"] = 5000.0 * 4.184
    native.setdefault("grid_width", 1001)
    native.setdefault("bias_width", 750.0)
    native["height_kj_mol"] = min(float(native.get("height_kj_mol", 1.0)), 0.2)
    native.setdefault("bias_factor", 15.0)
    native.setdefault("frequency", 1000)
    native.setdefault("save_frequency", 10000)
    native.setdefault("bias_dir", "native_bias")
    native["restart"] = False
    _set_output_dir(metad, str(output_dir / "metad"))
    metad_path = config_dir / "qregion_gap_table_metad.yaml"
    _write_yaml(metad_path, metad)

    legacy = _find_legacy_configs(base_config_path, config_dir)
    summary = {
        "selected_candidate_config": str(selected_config),
        "qregion_gap_umbrella": str(umbrella_path),
        "qregion_gap_table_metad": str(metad_path),
        "legacy_configs": legacy,
        "selected_candidate": SELECTED_CANDIDATE,
        "delta_alpha_kj_mol": SELECTED_DELTA_ALPHA,
        "h12_kj_mol": SELECTED_H12,
    }
    write_json(config_dir / "config_generation_summary.json", summary)
    return summary


def json_roundtrip(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _find_legacy_configs(base_config_path: str | Path, config_dir: Path) -> dict[str, str | None]:
    base = _read_yaml(base_config_path)
    legacy_metad = config_dir / "legacy_reference_gap_metad.yaml"
    _write_yaml(legacy_metad, base)
    legacy_umbrella = None
    for candidate in [Path("examples/hg317_evb_gap_umbrella.yaml"), Path("examples/hg317_irc_mapping_barrier_calibrated.yaml")]:
        if candidate.exists():
            legacy_umbrella = config_dir / "legacy_reference_gap_umbrella.yaml"
            shutil.copyfile(candidate, legacy_umbrella)
            break
    return {"legacy_reference_gap_metad": str(legacy_metad), "legacy_reference_gap_umbrella": None if legacy_umbrella is None else str(legacy_umbrella)}


def write_reproduction_scripts(output: str | Path, template_dir: str | Path = "scripts/run_hg317_qregion_reproduction") -> dict[str, Any]:
    output_dir = ensure_output_dir(Path(output) / "run_commands")
    template = ensure_output_dir(template_dir)
    scripts = {}
    script_specs = {
        "run_qregion_umbrella.sh": "umbrella",
        "run_qregion_metad.sh": "metad",
        "run_all_qregion_reproduction.sh": "all",
        "monitor_reproduction.sh": "monitor",
        "resume_reproduction.sh": "resume",
    }
    for name, workflow in script_specs.items():
        text = _script_text(workflow)
        for directory in [output_dir, template]:
            path = directory / name
            path.write_text(text, encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        scripts[name] = {"output_copy": str(output_dir / name), "template_copy": str(template / name)}
    return scripts


def _script_text(workflow: str) -> str:
    if workflow == "monitor":
        return """#!/usr/bin/env bash\nset -euo pipefail\nOUT=${1:-outputs/hg317_qregion_reproduction}\nfind \"$OUT\" -maxdepth 3 -name '*summary*.json' -o -name '*.log'\n"""
    if workflow == "resume":
        workflow = "all"
        extra = " --resume --skip-existing"
    else:
        extra = ""
    if workflow == "all":
        wf = "all"
    elif workflow == "umbrella":
        wf = "umbrella"
    elif workflow == "metad":
        wf = "metad"
    else:
        wf = workflow
    return f"""#!/usr/bin/env bash\nset -euo pipefail\nOUTPUT=outputs/hg317_qregion_reproduction\nPLATFORM=CUDA\nMODE=production\nREPLICAS=3\nUMBRELLA_STEPS=200000\nMETAD_STEPS=1000000\nwhile [[ $# -gt 0 ]]; do\n  case \"$1\" in\n    --output) OUTPUT=\"$2\"; shift 2;;\n    --platform) PLATFORM=\"$2\"; shift 2;;\n    --mode) MODE=\"$2\"; shift 2;;\n    --replicas) REPLICAS=\"$2\"; shift 2;;\n    --umbrella-steps) UMBRELLA_STEPS=\"$2\"; shift 2;;\n    --metad-steps) METAD_STEPS=\"$2\"; shift 2;;\n    --resume|--skip-existing) shift;;\n    *) echo \"unknown argument: $1\" >&2; exit 2;;\n  esac\ndone\nmkdir -p \"$OUTPUT/logs\"\necho \"git_sha=$(git rev-parse HEAD 2>/dev/null || true)\" | tee -a \"$OUTPUT/logs/run.log\"\necho \"platform=$PLATFORM mode=$MODE workflow={wf}\" | tee -a \"$OUTPUT/logs/run.log\"\n.venv/bin/evb hg317-qregion-reproduce \\\n  --config examples/hg317_evb_gap_metad.yaml \\\n  --reference examples/hg317_gxtb_reference_profile.yaml \\\n  --output \"$OUTPUT\" \\\n  --platform \"$PLATFORM\" \\\n  --mode \"$MODE\" \\\n  --workflow {wf} \\\n  --replicas \"$REPLICAS\" \\\n  --umbrella-steps \"$UMBRELLA_STEPS\" \\\n  --metad-steps \"$METAD_STEPS\"{extra} 2>&1 | tee -a \"$OUTPUT/logs/run.log\"\n"""


def run_reproduction_workflow(
    config: EVBConfig,
    config_path: str | Path,
    reference_path: str | Path,
    output: str | Path,
    *,
    platform: str | None = None,
    mode: str = "quick",
    workflow: str = "all",
    replicas: int | None = None,
    umbrella_steps: int | None = None,
    metad_steps: int | None = None,
    write_run_scripts_only: bool = False,
    resume: bool = False,
    skip_existing: bool = False,
    barrier_warning_kcal: float = 2.0,
    barrier_fail_kcal: float = 3.0,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    settings = settings_for_mode(mode, replicas, umbrella_steps, metad_steps)
    selected = prepare_selected_candidate_config(config_path, output_dir)
    config_summary = generate_reproduction_configs(config_path, selected, output_dir, settings)
    scripts = write_reproduction_scripts(output_dir)
    if write_run_scripts_only:
        summary = {"status": "scripts_only", "configs": config_summary, "run_scripts": scripts}
        write_json(output_dir / "workflow_summary.json", summary)
        return summary

    results: dict[str, Any] = {"configs": config_summary, "run_scripts": scripts, "settings": asdict(settings)}
    if workflow in {"umbrella", "all"}:
        results["umbrella"] = run_qregion_umbrella(config_summary["qregion_gap_umbrella"], reference_path, output_dir / "umbrella", platform, settings.umbrella_steps, settings.n_windows, skip_existing)
    if workflow in {"metad", "all"}:
        results["metad"] = run_qregion_metad(config_summary["qregion_gap_table_metad"], reference_path, output_dir / "metad", platform, settings.metad_steps, settings.replicas, skip_existing)
    if workflow in {"analysis-only", "analysis", "all", "umbrella", "metad"}:
        results["comparison"] = analyze_and_compare_reproduction(output_dir, reference_path, barrier_warning_kcal, barrier_fail_kcal)
    write_json(output_dir / "workflow_summary.json", results)
    return results


def run_qregion_umbrella(config_path: str | Path, reference_path: str | Path, output: str | Path, platform: str | None, steps: int, n_windows: int, skip_existing: bool = False) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    if skip_existing and (output_dir / "analysis" / "barrier_summary.json").exists():
        return json.loads((output_dir / "analysis" / "barrier_summary.json").read_text(encoding="utf-8"))
    # Quick executable approximation: run a short unbiased Q-region trajectory once and analyze its gap histogram.
    stability = run_hg317_qregion_stability_check(config_path, output_dir / "windows" / "w000", platform, plain_steps=steps, table_metad_steps=0, quick=False, report_interval=max(1, min(1000, steps)))
    obs = output_dir / "windows" / "w000" / "plain_md" / "observables.csv"
    analysis_dir = ensure_output_dir(output_dir / "analysis")
    pmf = pmf_from_observables([obs], analysis_dir / "pmf.csv")
    barrier = calculate_barrier_from_pmf(pmf["gap_kcal_mol"], pmf["pmf_kcal_mol"])
    barrier.update({"method": "histogram_fallback", "steps": steps, "n_windows_requested": n_windows, "stability": _compact_run(stability["runs"][0])})
    write_json(analysis_dir / "barrier_summary.json", barrier)
    plot_pmf(analysis_dir / "pmf.csv", analysis_dir / "qregion_gap_umbrella_pmf.png", "Q-region gap umbrella quick PMF")
    shutil.copyfile(analysis_dir / "qregion_gap_umbrella_pmf.png", output_dir / "qregion_gap_umbrella_pmf.png")
    return barrier


def run_qregion_metad(config_path: str | Path, reference_path: str | Path, output: str | Path, platform: str | None, steps: int, replicas: int, skip_existing: bool = False) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    rep_summaries = []
    colvars = []
    for rep in range(1, replicas + 1):
        rep_dir = output_dir / f"rep{rep:02d}"
        if skip_existing and (rep_dir / "summary.json").exists():
            rep_summaries.append(json.loads((rep_dir / "summary.json").read_text(encoding="utf-8")))
            colvars.append(rep_dir / "colvar.csv")
            continue
        stability = run_hg317_qregion_stability_check(config_path, rep_dir / "stability", platform, plain_steps=0, table_metad_steps=steps, quick=False, report_interval=max(1, min(1000, steps)))
        src = rep_dir / "stability" / "table_metad" / "observables.csv"
        colvar = rep_dir / "colvar.csv"
        rep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, colvar)
        final_src = rep_dir / "stability" / "table_metad" / "final.pdb"
        if final_src.exists():
            shutil.copyfile(final_src, rep_dir / "final.pdb")
        bias_dir = rep_dir / "stability" / "table_metad" / "native_bias"
        for source, target in [
            (bias_dir / "bias_state.json", rep_dir / "native_gap_bias_state.json"),
            (bias_dir / "bias_table.csv", rep_dir / "native_gap_bias_table.csv"),
        ]:
            if source.exists():
                shutil.copyfile(source, target)
        summary = _compact_run(stability["runs"][0])
        write_json(rep_dir / "summary.json", summary)
        rep_summaries.append(summary)
        colvars.append(colvar)
    analysis_dir = ensure_output_dir(output_dir / "analysis")
    metad = analyze_metad_replicas(colvars, analysis_dir)
    metad["replicas"] = rep_summaries
    write_json(analysis_dir / "barrier_summary.json", metad["barrier_summary"])
    return metad


def _compact_run(row: dict[str, Any]) -> dict[str, Any]:
    keys = ["mode", "stable", "sampling_promising", "completed_steps", "steps_per_s", "ns_per_day", "shifted_gap_min_kj_mol", "shifted_gap_max_kj_mol", "w2_min", "w2_max", "mixing_region_frame_count", "duplicated_full_nonbonded", "pme_approximation", "number_of_bias_updates", "average_time_per_bias_update_s"]
    return {key: row.get(key) for key in keys}


def _read_observable_gaps(paths: list[str | Path]) -> np.ndarray:
    values = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                gap = row.get("shifted_gap") or row.get("shifted_gap_kj_mol")
                if gap is not None and gap != "":
                    values.append(float(gap) * KJ_TO_KCAL)
    return np.asarray(values, dtype=float)


def pmf_from_observables(paths: list[str | Path], output_csv: str | Path, bins: int = 80) -> dict[str, np.ndarray]:
    gaps = _read_observable_gaps(paths)
    if len(gaps) == 0:
        grid = np.linspace(-1, 1, bins)
        pmf = np.full_like(grid, np.nan)
    else:
        hist, edges = np.histogram(gaps, bins=min(bins, max(5, len(gaps) // 2)))
        grid = 0.5 * (edges[:-1] + edges[1:])
        prob = np.maximum(hist.astype(float), 1.0e-12)
        pmf = -0.00198720425864083 * 300.0 * np.log(prob)
        pmf = pmf - np.nanmin(pmf)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kcal_mol", "pmf_kcal_mol"])
        for x, y in zip(grid, pmf):
            writer.writerow([float(x), float(y)])
    return {"gap_kcal_mol": grid, "pmf_kcal_mol": pmf}


def calculate_barrier_from_pmf(gap_kcal_mol: np.ndarray | list[float], pmf_kcal_mol: np.ndarray | list[float]) -> dict[str, float | None]:
    gap = np.asarray(gap_kcal_mol, dtype=float)
    pmf = np.asarray(pmf_kcal_mol, dtype=float)
    mask = np.isfinite(gap) & np.isfinite(pmf)
    gap = gap[mask]
    pmf = pmf[mask]
    if len(gap) == 0:
        return {"barrier_from_left_kcal": None, "barrier_from_right_kcal": None, "deltaG_reaction_kcal": None, "pmf_at_gap0_kcal": None}
    left = gap <= 0
    right = gap >= 0
    left_idx = int(np.argmin(pmf[left])) if np.any(left) else int(np.argmin(pmf))
    right_idx = int(np.argmin(pmf[right])) if np.any(right) else int(np.argmin(pmf))
    left_gap = gap[left][left_idx] if np.any(left) else gap[left_idx]
    left_min = pmf[left][left_idx] if np.any(left) else pmf[left_idx]
    right_gap = gap[right][right_idx] if np.any(right) else gap[right_idx]
    right_min = pmf[right][right_idx] if np.any(right) else pmf[right_idx]
    order = np.argsort(gap)
    pmf0 = float(np.interp(0.0, gap[order], pmf[order])) if len(gap) > 1 else float(pmf[0])
    return {
        "left_min_gap_kcal": float(left_gap),
        "right_min_gap_kcal": float(right_gap),
        "barrier_from_left_kcal": float(pmf0 - left_min),
        "barrier_from_right_kcal": float(pmf0 - right_min),
        "deltaG_reaction_kcal": float(right_min - left_min),
        "pmf_at_gap0_kcal": pmf0,
    }


def analyze_metad_replicas(colvar_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    replica_curves = []
    grids = []
    pmfs = []
    for idx, path in enumerate(colvar_paths, start=1):
        pmf = pmf_from_observables([path], output_dir / f"rep{idx:02d}_fel.csv")
        replica_curves.append({"replica": idx, "path": str(path), "n_points": int(len(pmf["gap_kcal_mol"]))})
        grids.append(pmf["gap_kcal_mol"])
        pmfs.append(pmf["pmf_kcal_mol"])
    if pmfs:
        common = np.linspace(min(float(np.nanmin(g)) for g in grids), max(float(np.nanmax(g)) for g in grids), 100)
        interp = np.vstack([np.interp(common, g, p) for g, p in zip(grids, pmfs)])
        mean = np.nanmean(interp, axis=0)
        std = np.nanstd(interp, axis=0)
    else:
        common = np.linspace(-1, 1, 100)
        mean = np.full_like(common, np.nan)
        std = np.full_like(common, np.nan)
    mean = mean - np.nanmin(mean) if np.any(np.isfinite(mean)) else mean
    with (output_dir / "qregion_gap_metad_fel.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kcal_mol", "mean_fel_kcal_mol", "sd_fel_kcal_mol"])
        for x, y, s in zip(common, mean, std):
            writer.writerow([float(x), float(y), float(s)])
    barrier = calculate_barrier_from_pmf(common, mean)
    plot_pmf(output_dir / "qregion_gap_metad_fel.csv", output_dir / "qregion_gap_metad_hist_fel.png", "Q-region gap metad FEL", y_col="mean_fel_kcal_mol")
    plot_pmf(output_dir / "qregion_gap_metad_fel.csv", output_dir / "qregion_gap_metad_replicas_pmf.png", "Q-region gap metad replicas PMF", y_col="mean_fel_kcal_mol")
    return {"replica_curves": replica_curves, "barrier_summary": barrier, "mean_fel_csv": str(output_dir / "qregion_gap_metad_fel.csv")}


def plot_pmf(csv_path: str | Path, png_path: str | Path, title: str, y_col: str = "pmf_kcal_mol") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        Path(png_path).write_text("matplotlib unavailable\n", encoding="utf-8")
        return
    xs, ys = [], []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["gap_kcal_mol"]))
            ys.append(float(row[y_col]))
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, lw=1.8)
    plt.axvline(0.0, color="k", ls="--", lw=1)
    plt.xlabel("shifted EVB gap (kcal/mol)")
    plt.ylabel("PMF / FEL (kcal/mol)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def analyze_and_compare_reproduction(output: str | Path, reference_path: str | Path, warning_kcal: float, fail_kcal: float) -> dict[str, Any]:
    output_dir = ensure_output_dir(output)
    comparison_dir = ensure_output_dir(output_dir / "comparison")
    reference = load_gxtb_reference_profile(reference_path, "relative_to_RC")
    gxtb_barrier = reference.target_kcal_mol.get("TS", 18.13981382)
    rows = [{"method": "g-xTB RC->TS", "barrier_kcal_mol": gxtb_barrier, "delta_vs_gxtb": 0.0, "delta_vs_legacy": None, "status": "reference"}]
    qregion = {}
    for method, path in [
        ("q_region umbrella", output_dir / "umbrella" / "analysis" / "barrier_summary.json"),
        ("q_region table-metad", output_dir / "metad" / "analysis" / "barrier_summary.json"),
    ]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            barrier = data.get("barrier_from_left_kcal")
            delta = None if barrier is None else float(barrier) - float(gxtb_barrier)
            status = "missing" if barrier is None else ("fail" if abs(delta) > fail_kcal else "warning" if abs(delta) > warning_kcal else "pass")
            rows.append({"method": method, "barrier_kcal_mol": barrier, "delta_vs_gxtb": delta, "delta_vs_legacy": None, "status": status})
            qregion[method] = data
    legacy = find_legacy_reference_summary()
    speed = compute_speedup_summary(output_dir, legacy)
    _write_barrier_comparison(comparison_dir, rows)
    write_json(comparison_dir / "legacy_reference_summary.json", legacy)
    write_json(comparison_dir / "qregion_summary.json", qregion)
    write_json(comparison_dir / "speedup_summary.json", speed)
    report = {"barriers": rows, "legacy": legacy, "qregion": qregion, "speedup": speed, "warning_kcal": warning_kcal, "fail_kcal": fail_kcal}
    write_json(comparison_dir / "comparison_summary.json", report)
    _write_reproduction_report(comparison_dir / "reproduction_report.md", report)
    return report


def find_legacy_reference_summary() -> dict[str, Any]:
    candidates = [
        Path("outputs/hg317_evb_gap_metad/summary/barrier_summary.json"),
        Path("outputs/hg317_evb_gap_metad/analysis/barrier_summary.json"),
        Path("outputs/hg317_evb_gap_umbrella/analysis/barrier_summary.json"),
    ]
    for path in candidates:
        if path.exists():
            return {"available": True, "path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    return {"available": False, "reason": "Previous full-state numeric PMF/metad JSON/CSV outputs were not found; PNGs were not parsed."}


def compute_speedup_summary(output_dir: Path, legacy: dict[str, Any] | None = None) -> dict[str, Any]:
    methods = []
    for name, path in [
        ("q_region umbrella", output_dir / "umbrella" / "windows" / "w000" / "plain_md" / "stability_summary.json"),
        ("q_region table-metad", output_dir / "metad" / "rep01" / "stability" / "table_metad" / "stability_summary.json"),
    ]:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            methods.append({"method": name, "steps_per_s": data.get("steps_per_s"), "ns_per_day": data.get("ns_per_day"), "duplicated_full_nonbonded": data.get("duplicated_full_nonbonded"), "pme_approximation": data.get("pme_approximation")})
    return {"legacy_reference_available": bool(legacy and legacy.get("available")), "methods": methods, "speedup_vs_legacy": None}


def _write_barrier_comparison(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "barrier_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "barrier_kcal_mol", "delta_vs_gxtb", "delta_vs_legacy", "status"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "barrier_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("| method | barrier kcal/mol | delta vs g-xTB | delta vs legacy | status |\n")
        handle.write("| --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(f"| {row['method']} | {row.get('barrier_kcal_mol')} | {row.get('delta_vs_gxtb')} | {row.get('delta_vs_legacy')} | {row.get('status')} |\n")


def _write_reproduction_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# HG3.17 Q-region Reproduction Report\n\n")
        handle.write("No OPES was implemented in this workflow.\n\n")
        handle.write("## Barrier comparison\n\n")
        handle.write("| method | barrier kcal/mol | delta vs g-xTB | status |\n")
        handle.write("| --- | ---: | ---: | --- |\n")
        for row in report["barriers"]:
            handle.write(f"| {row['method']} | {row.get('barrier_kcal_mol')} | {row.get('delta_vs_gxtb')} | {row.get('status')} |\n")
        handle.write("\n## Legacy reference\n\n")
        handle.write(json.dumps(report["legacy"], indent=2))

