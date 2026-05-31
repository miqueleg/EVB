from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import EVBConfig
from ..evb import EVBHamiltonian, EVBParameters


@dataclass(slots=True)
class CouplingScanCurve:
    h12_kj_mol: float
    rc_centers: list[float]
    mean_evb_energy_kj_mol: list[float]
    counts: list[int]


def load_frames_for_coupling_scan(output_dir: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for csv_path in sorted((Path(output_dir) / "windows").glob("w*/production_observables.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                donor_h = float(row["donor_h"]) if "donor_h" in row and row["donor_h"] else np.nan
                h_acceptor = float(row["h_acceptor"]) if "h_acceptor" in row and row["h_acceptor"] else np.nan
                rc = donor_h - h_acceptor if np.isfinite(donor_h) and np.isfinite(h_acceptor) else float(row["delta_e_shifted_kj_mol"])
                rows.append(
                    {
                        "window_lambda": float(row["lambda"]),
                        "E1": float(row["E1_kj_mol"]),
                        "E2": float(row["E2_kj_mol"]),
                        "rc": rc,
                        "delta_e_shifted": float(row["delta_e_shifted_kj_mol"]),
                    }
                )
    if not rows:
        raise ValueError(f"No sampled frames found under {Path(output_dir) / 'windows'}")
    return rows


def build_coupling_scan(
    config: EVBConfig,
    h12_values: list[float],
    delta_alpha: float | None = None,
    n_bins: int = 40,
) -> list[CouplingScanCurve]:
    frames = load_frames_for_coupling_scan(config.output_dir)
    rc_values = np.asarray([frame["rc"] for frame in frames], dtype=float)
    e1_values = np.asarray([frame["E1"] for frame in frames], dtype=float)
    e2_values = np.asarray([frame["E2"] for frame in frames], dtype=float)
    if delta_alpha is None:
        delta_alpha = config.evb_parameters.delta_alpha or 0.0
    rc_min = float(np.min(rc_values))
    rc_max = float(np.max(rc_values))
    if np.isclose(rc_min, rc_max):
        rc_min -= 0.01
        rc_max += 0.01
    edges = np.linspace(rc_min, rc_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    curves: list[CouplingScanCurve] = []
    for h12 in h12_values:
        ham = EVBHamiltonian(EVBParameters(delta_alpha=delta_alpha, h12=h12))
        evb_energies = np.asarray([ham.lower_eigenvalue(float(e1), float(e2))[0] for e1, e2 in zip(e1_values, e2_values)], dtype=float)
        mean_values: list[float] = []
        counts: list[int] = []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (rc_values >= left) & (rc_values < right)
            if right == edges[-1]:
                mask = (rc_values >= left) & (rc_values <= right)
            count = int(np.count_nonzero(mask))
            counts.append(count)
            mean_values.append(float(np.mean(evb_energies[mask])) if count else np.nan)
        curves.append(
            CouplingScanCurve(
                h12_kj_mol=float(h12),
                rc_centers=centers.astype(float).tolist(),
                mean_evb_energy_kj_mol=mean_values,
                counts=counts,
            )
        )
    return curves


def write_coupling_scan_outputs(
    output_dir: str | Path,
    curves: list[CouplingScanCurve],
    *,
    rc_label: str = "donor_h_minus_h_acceptor_nm",
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "coupling_scan.csv"
    png_path = output_dir / "coupling_scan.png"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["h12_kj_mol", "rc", "mean_evb_energy_kj_mol", "counts"])
        for curve in curves:
            for rc, energy, count in zip(curve.rc_centers, curve.mean_evb_energy_kj_mol, curve.counts):
                writer.writerow([curve.h12_kj_mol, rc, energy, count])

    plt.figure(figsize=(8, 5))
    for curve in curves:
        x = np.asarray(curve.rc_centers, dtype=float)
        y = np.asarray(curve.mean_evb_energy_kj_mol, dtype=float)
        mask = np.isfinite(y)
        if np.any(mask):
            plt.plot(x[mask], y[mask], marker="o", linewidth=1.5, markersize=3, label=f"H12={curve.h12_kj_mol:g}")
    plt.xlabel(rc_label)
    plt.ylabel("Mean EVB Energy (kJ/mol)")
    plt.title("EVB Energy vs Reaction Coordinate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    return csv_path, png_path
