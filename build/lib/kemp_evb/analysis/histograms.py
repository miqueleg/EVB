from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import EVBConfig


@dataclass(slots=True)
class WindowObservableData:
    window_id: str
    lambda_value: float | None
    gap_center_kj_mol: float | None
    values_kj_mol: np.ndarray
    n_frames: int
    energy1_kj_mol: np.ndarray | None = None
    energy2_kj_mol: np.ndarray | None = None
    evb_energy_kj_mol: np.ndarray | None = None


@dataclass(slots=True)
class HistogramResult:
    window_id: str
    lambda_value: float | None
    gap_center_kj_mol: float | None
    counts: list[int]
    probabilities: list[float]


def load_window_observables(output_dir: str | Path, shifted: bool = True) -> list[WindowObservableData]:
    root = Path(output_dir) / "windows"
    if not root.exists():
        raise FileNotFoundError(f"No windows directory found at {root}")
    column = "delta_e_shifted_kj_mol" if shifted else "delta_e_kj_mol"
    results: list[WindowObservableData] = []
    for csv_path in sorted(root.glob("*/production_observables.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
        lambda_value = float(rows[0]["lambda"]) if "lambda" in rows[0] and rows[0]["lambda"] else None
        gap_center = float(rows[0]["gap_center_kj_mol"]) if "gap_center_kj_mol" in rows[0] and rows[0]["gap_center_kj_mol"] else None
        energy1 = (
            np.asarray([float(row["E1_kj_mol"]) for row in rows], dtype=float)
            if "E1_kj_mol" in rows[0]
            else None
        )
        energy2 = (
            np.asarray([float(row["E2_kj_mol"]) for row in rows], dtype=float)
            if "E2_kj_mol" in rows[0]
            else None
        )
        evb_energy = (
            np.asarray([float(row["Eevb_kj_mol"]) for row in rows], dtype=float)
            if "Eevb_kj_mol" in rows[0]
            else None
        )
        results.append(
            WindowObservableData(
                window_id=rows[0]["window_id"],
                lambda_value=lambda_value,
                gap_center_kj_mol=gap_center,
                values_kj_mol=values,
                n_frames=len(rows),
                energy1_kj_mol=energy1,
                energy2_kj_mol=energy2,
                evb_energy_kj_mol=evb_energy,
            )
        )
    if not results:
        raise ValueError(f"No production_observables.csv files with data were found under {root}")
    return results


def histogram_edges(config: EVBConfig) -> np.ndarray:
    spec = config.analysis.histogram
    return np.linspace(spec.bin_min_kj_mol, spec.bin_max_kj_mol, spec.n_bins + 1)


def histogram_centers(config: EVBConfig) -> np.ndarray:
    edges = histogram_edges(config)
    return 0.5 * (edges[:-1] + edges[1:])


def build_gap_histograms(config: EVBConfig, window_data: list[WindowObservableData]) -> tuple[np.ndarray, list[HistogramResult]]:
    edges = histogram_edges(config)
    results: list[HistogramResult] = []
    for window in window_data:
        counts, _ = np.histogram(window.values_kj_mol, bins=edges)
        total = counts.sum()
        probabilities = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
        results.append(
            HistogramResult(
                window_id=window.window_id,
                lambda_value=window.lambda_value,
                gap_center_kj_mol=window.gap_center_kj_mol,
                counts=counts.astype(int).tolist(),
                probabilities=probabilities.astype(float).tolist(),
            )
        )
    return edges, results
