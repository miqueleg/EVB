from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evb import EVBHamiltonian, EVBParameters
from .reference_profile import ReferenceProfile, load_reference_profile


@dataclass(slots=True)
class CalibrationFrameEnergy:
    label: str
    e1_kj_mol: float
    e2_kj_mol: float


@dataclass(slots=True)
class CalibrationResult:
    delta_alpha_kj_mol: float
    h12_kj_mol: float
    rms_residual_kj_mol: float
    max_residual_kj_mol: float
    limited_fit: bool
    profile_rows: list[dict[str, float | str]]


def evb_lower(e1: float, e2: float, delta_alpha: float, h12: float) -> float:
    return EVBHamiltonian(EVBParameters(delta_alpha, h12)).lower_eigenvalue(e1, e2)[0]


def fit_profile_parameters(
    frame_energies: list[CalibrationFrameEnergy],
    reference: ReferenceProfile,
    *,
    initial_delta_alpha: float = 0.0,
    initial_h12: float = 100.0,
    fit_delta_alpha: bool = True,
    fit_h12: bool = True,
    mode: str = "profile_fit",
) -> CalibrationResult:
    targets = reference.target_kj_mol
    labels = [f.label for f in frame_energies if f.label in targets]
    if len(labels) < 2:
        raise ValueError("At least two frame energies with reference targets are required for calibration.")
    limited = mode == "barrier_only_fit" or "product" not in labels
    frames = [f for f in frame_energies if f.label in labels]
    delta_center = float(initial_delta_alpha)
    raw_gaps = [f.e1_kj_mol - f.e2_kj_mol for f in frames]
    if fit_delta_alpha and raw_gaps:
        delta_center = float(np.median(raw_gaps))
    delta_values = [delta_center] if not fit_delta_alpha else np.linspace(delta_center - 50000.0, delta_center + 50000.0, 401)
    hmax = max(2000.0, abs(initial_h12) * 2.0)
    h_values = [float(initial_h12)] if not fit_h12 else np.linspace(0.0, hmax, 201)
    best = _scan(frames, targets, delta_values, h_values)
    delta_values = np.linspace(best[0] - 1000.0, best[0] + 1000.0, 401) if fit_delta_alpha else [best[0]]
    h_values = np.linspace(max(0.0, best[1] - 200.0), best[1] + 200.0, 201) if fit_h12 else [best[1]]
    best = _scan(frames, targets, delta_values, h_values)
    rows, rms, max_abs = _profile_rows(frames, targets, best[0], best[1])
    return CalibrationResult(float(best[0]), float(best[1]), rms, max_abs, limited, rows)


def _scan(frames, targets, deltas, hs):
    best = (float("inf"), None, None)
    for delta in deltas:
        for h12 in hs:
            _, rms, _ = _profile_rows(frames, targets, float(delta), float(h12))
            if rms < best[0]:
                best = (rms, float(delta), float(h12))
    return best[1], best[2], best[0]


def _profile_rows(frames, targets, delta, h12):
    energies = {f.label: evb_lower(f.e1_kj_mol, f.e2_kj_mol, delta, h12) for f in frames}
    zero_label = frames[0].label
    if "reactant" in energies:
        zero_label = "reactant"
    zero = energies[zero_label]
    rows = []
    residuals = []
    for f in frames:
        model = energies[f.label] - zero
        target = targets[f.label]
        residual = model - target
        residuals.append(residual)
        rows.append({"label": f.label, "target_kj_mol": target, "model_kj_mol": model, "residual_kj_mol": residual, "raw_gap_kj_mol": f.e1_kj_mol - f.e2_kj_mol, "shifted_gap_kj_mol": f.e1_kj_mol - f.e2_kj_mol - delta})
    arr = np.asarray(residuals, dtype=float)
    return rows, float(np.sqrt(np.mean(arr * arr))), float(np.max(np.abs(arr)))


def load_frame_energies(path: str | Path) -> list[CalibrationFrameEnergy]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(CalibrationFrameEnergy(row["label"], float(row["e1_kj_mol"]), float(row["e2_kj_mol"])))
    return rows


def write_calibration_outputs(result: CalibrationResult, output: str | Path) -> dict[str, str]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fitted_parameters.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    with (out / "profile_fit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.profile_rows[0]))
        writer.writeheader(); writer.writerows(result.profile_rows)
    return {"parameters": str(out / "fitted_parameters.json"), "profile_fit": str(out / "profile_fit.csv")}


def calibrate_profile_from_files(reference_path: str | Path, frame_energy_csv: str | Path, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    reference = load_reference_profile(reference_path)
    result = fit_profile_parameters(load_frame_energies(frame_energy_csv), reference, **kwargs)
    files = write_calibration_outputs(result, output)
    return {"result": asdict(result), "files": files}
