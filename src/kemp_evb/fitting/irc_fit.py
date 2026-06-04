from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..evb import EVBHamiltonian, EVBParameters


@dataclass(slots=True)
class IRCFitResult:
    delta_alpha_kj_mol: float
    h12_kj_mol: float
    objective_rmse_kj_mol: float
    reaction_energy_kj_mol: float
    barrier_kj_mol: float
    n_points: int


def fit_evb_to_irc_profile(
    e1_kj_mol: np.ndarray,
    e2_kj_mol: np.ndarray,
    qm_relative_kj_mol: np.ndarray,
    delta_alpha_initial_kj_mol: float | None = None,
    h12_initial_kj_mol: float | None = None,
    levels: int = 7,
    samples_per_axis: int = 81,
) -> tuple[IRCFitResult, np.ndarray]:
    """Fit constant-H12 EVB parameters to a whole IRC profile by grid refinement."""
    e1 = np.asarray(e1_kj_mol, dtype=float)
    e2 = np.asarray(e2_kj_mol, dtype=float)
    target = np.asarray(qm_relative_kj_mol, dtype=float)
    if not (len(e1) == len(e2) == len(target)):
        raise ValueError("e1, e2, and qm_relative arrays must have the same length.")
    if len(e1) < 3:
        raise ValueError("At least three IRC points are required for EVB fitting.")

    finite = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(target)
    e1 = e1[finite]
    e2 = e2[finite]
    target = target[finite]
    target = target - target[0]

    raw_gap = e1 - e2
    delta_center = float(np.median(raw_gap)) if delta_alpha_initial_kj_mol is None else float(delta_alpha_initial_kj_mol)
    target_span = max(float(np.nanmax(target) - np.nanmin(target)), 100.0)
    gap_span = max(float(np.nanmax(raw_gap) - np.nanmin(raw_gap)), target_span, 1000.0)
    h12_center = 0.5 * gap_span if h12_initial_kj_mol is None else float(h12_initial_kj_mol)
    delta_lo = delta_center - gap_span
    delta_hi = delta_center + gap_span
    h12_lo = max(0.0, h12_center - gap_span)
    h12_hi = h12_center + gap_span
    best_result: IRCFitResult | None = None
    best_profile: np.ndarray | None = None

    for _ in range(levels):
        for delta_alpha in np.linspace(delta_lo, delta_hi, samples_per_axis):
            for h12 in np.linspace(h12_lo, h12_hi, samples_per_axis):
                profile = evb_relative_profile(e1, e2, float(delta_alpha), float(h12))
                residual = profile - target
                rmse = float(np.sqrt(np.mean(residual * residual)))
                if best_result is None or rmse < best_result.objective_rmse_kj_mol:
                    best_result = _build_result(float(delta_alpha), float(h12), rmse, profile)
                    best_profile = profile
        assert best_result is not None
        assert best_profile is not None
        delta_width = max((delta_hi - delta_lo) / 8.0, 1.0e-3)
        h12_width = max((h12_hi - h12_lo) / 8.0, 1.0e-3)
        delta_lo = best_result.delta_alpha_kj_mol - delta_width
        delta_hi = best_result.delta_alpha_kj_mol + delta_width
        h12_lo = max(0.0, best_result.h12_kj_mol - h12_width)
        h12_hi = best_result.h12_kj_mol + h12_width

    assert best_result is not None
    assert best_profile is not None
    return best_result, best_profile


def evb_relative_profile(e1_kj_mol: np.ndarray, e2_kj_mol: np.ndarray, delta_alpha_kj_mol: float, h12_kj_mol: float) -> np.ndarray:
    hamiltonian = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha_kj_mol, h12=h12_kj_mol))
    values = np.asarray([hamiltonian.lower_eigenvalue(float(e1), float(e2))[0] for e1, e2 in zip(e1_kj_mol, e2_kj_mol)], dtype=float)
    return values - values[0]


def _build_result(delta_alpha: float, h12: float, rmse: float, profile: np.ndarray) -> IRCFitResult:
    return IRCFitResult(
        delta_alpha_kj_mol=delta_alpha,
        h12_kj_mol=h12,
        objective_rmse_kj_mol=rmse,
        reaction_energy_kj_mol=float(profile[-1] - profile[0]),
        barrier_kj_mol=float(np.nanmax(profile) - profile[0]),
        n_points=len(profile),
    )


def fit_irc_energy_table(table_path: str | Path, output_dir: str | Path) -> IRCFitResult:
    frames = []
    e1 = []
    e2 = []
    qm = []
    with Path(table_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frames.append(int(row["frame"]))
            e1.append(float(row["E1_kj_mol"]))
            e2.append(float(row["E2_kj_mol"]))
            qm.append(float(row["qm_relative_kj_mol"]))
    result, profile = fit_evb_to_irc_profile(np.asarray(e1), np.asarray(e2), np.asarray(qm))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "irc_evb_fit.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    with (destination / "irc_evb_fit_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "qm_relative_kj_mol", "evb_relative_kj_mol", "residual_kj_mol"])
        for frame, qm_value, evb_value in zip(frames, qm, profile):
            writer.writerow([frame, qm_value, evb_value, evb_value - qm_value])
    return result
