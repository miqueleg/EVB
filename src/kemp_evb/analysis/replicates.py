from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import write_json


@dataclass(slots=True)
class ReplicateBarrierSummary:
    n_replicates: int
    barrier_mean_kj_mol: float | None
    barrier_std_kj_mol: float | None
    reaction_free_energy_mean_kj_mol: float | None
    reaction_free_energy_std_kj_mol: float | None


def summarize_replicates(output_dirs: list[str | Path], destination: str | Path) -> dict:
    if not output_dirs:
        raise ValueError("At least one replicate output directory is required.")
    output_paths = [Path(path) for path in output_dirs]
    pmf_tables = [_load_pmf_table(path / "analysis" / "pmf_gap.csv") for path in output_paths]
    gap_axis = pmf_tables[0]["gap_kj_mol"]
    for table in pmf_tables[1:]:
        if table["gap_kj_mol"] != gap_axis:
            raise ValueError("Replicate PMF grids do not match; use the same histogram settings for all replicates.")

    pmf_matrix = np.asarray([table["free_energy_kj_mol"] for table in pmf_tables], dtype=float)
    mean_pmf = np.nanmean(pmf_matrix, axis=0)
    std_pmf = np.nanstd(pmf_matrix, axis=0, ddof=1) if len(pmf_tables) > 1 else np.zeros_like(mean_pmf)
    finite_mask = np.isfinite(mean_pmf)
    if np.any(finite_mask):
        mean_pmf[finite_mask] -= np.nanmin(mean_pmf[finite_mask])

    barrier_values = []
    reaction_values = []
    for path in output_paths:
        with (path / "analysis" / "barrier_estimate.json").open("r", encoding="utf-8") as handle:
            barrier = json.load(handle)
        if barrier.get("barrier_forward_kj_mol") is not None:
            barrier_values.append(float(barrier["barrier_forward_kj_mol"]))
        if barrier.get("reaction_free_energy_kj_mol") is not None:
            reaction_values.append(float(barrier["reaction_free_energy_kj_mol"]))

    summary = ReplicateBarrierSummary(
        n_replicates=len(output_paths),
        barrier_mean_kj_mol=float(np.mean(barrier_values)) if barrier_values else None,
        barrier_std_kj_mol=float(np.std(barrier_values, ddof=1)) if len(barrier_values) > 1 else 0.0 if barrier_values else None,
        reaction_free_energy_mean_kj_mol=float(np.mean(reaction_values)) if reaction_values else None,
        reaction_free_energy_std_kj_mol=float(np.std(reaction_values, ddof=1)) if len(reaction_values) > 1 else 0.0 if reaction_values else None,
    )

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_replicate_pmf_csv(destination / "pmf_gap_replicates.csv", gap_axis, mean_pmf, std_pmf)
    _write_replicate_plot(destination / "pmf_gap_replicates.png", gap_axis, mean_pmf, std_pmf)
    write_json(destination / "replicate_summary.json", summary)
    payload = {
        "replicates": [str(path) for path in output_paths],
        "summary": summary,
    }
    return payload


def _load_pmf_table(path: Path) -> dict[str, list[float]]:
    gap = []
    free_energy = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gap.append(float(row["gap_kj_mol"]))
            value = row["free_energy_kj_mol"]
            free_energy.append(float(value) if value else np.nan)
    return {"gap_kj_mol": gap, "free_energy_kj_mol": free_energy}


def _write_replicate_pmf_csv(path: Path, gap_axis: list[float], mean_pmf: np.ndarray, std_pmf: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kj_mol", "mean_free_energy_kj_mol", "std_free_energy_kj_mol"])
        for gap, mean, std in zip(gap_axis, mean_pmf, std_pmf):
            writer.writerow([gap, "" if np.isnan(mean) else float(mean), "" if np.isnan(std) else float(std)])


def _write_replicate_plot(path: Path, gap_axis: list[float], mean_pmf: np.ndarray, std_pmf: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(gap_axis, dtype=float)
    y = np.asarray(mean_pmf, dtype=float)
    yerr = np.asarray(std_pmf, dtype=float)
    mask = np.isfinite(y)
    plt.figure(figsize=(6.2, 4.2), dpi=220)
    plt.plot(x[mask], y[mask], color="black", linewidth=1.8)
    if np.any(mask):
        plt.fill_between(x[mask], y[mask] - yerr[mask], y[mask] + yerr[mask], color="0.6", alpha=0.35, linewidth=0.0)
    plt.xlabel("EVB Energy Gap / kJ mol$^{-1}$")
    plt.ylabel("Free Energy / kJ mol$^{-1}$")
    plt.title("Kemp Solvent EVB Umbrella Replicates")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
