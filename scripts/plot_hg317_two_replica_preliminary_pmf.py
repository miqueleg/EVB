from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "hg317_evb_mapping_replicates"
REPORT_DIR = OUTPUT_ROOT / "current_report"
KJ_TO_KCAL = 1.0 / 4.184
R_KJ_MOL_K = 0.008314462618
TEMPERATURE_K = 300.0
DELTA_ALPHA_KJ_MOL = -53328.25448088075
FRAME_STRIDE = 4
MAX_MBAR_ITERATIONS = 1000


@dataclass(slots=True)
class WindowSamples:
    replicate: str
    window_id: str
    lambda_value: float
    gap_kj_mol: np.ndarray
    e1_kj_mol: np.ndarray
    e2_kj_mol: np.ndarray
    evb_kj_mol: np.ndarray


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    windows = _load_complete_core_windows()
    if not windows:
        raise SystemExit("No complete core windows found for rep01/rep02.")
    combined = _mbar_gap_pmf(windows)
    _write_pmf_csv(REPORT_DIR / "preliminary_two_replica_core_gap_pmf_kcal.csv", combined)
    _plot_pmf(REPORT_DIR / "preliminary_two_replica_core_gap_pmf_kcal.png", combined)

    per_rep = {}
    for rep in sorted({window.replicate for window in windows}):
        rep_windows = [window for window in windows if window.replicate == rep]
        per_rep[rep] = _mbar_gap_pmf(rep_windows, bin_edges_kj_mol=combined["bin_edges_kj_mol"])
    _plot_replicate_pmf(REPORT_DIR / "preliminary_two_replica_core_gap_pmf_by_replica_kcal.png", combined, per_rep)
    _write_summary(REPORT_DIR / "preliminary_two_replica_core_gap_pmf_summary.json", windows, combined, per_rep)
    print(f"Wrote preliminary two-replica PMF to {REPORT_DIR}")


def _load_complete_core_windows() -> list[WindowSamples]:
    windows: list[WindowSamples] = []
    for rep in ("rep01", "rep02"):
        rep_dir = OUTPUT_ROOT / rep / "windows"
        if not rep_dir.is_dir():
            continue
        for csv_path in sorted(rep_dir.glob("w*/production_observables.csv")):
            window_id = csv_path.parent.name
            window_index = int(window_id[1:])
            if window_index == 0 or window_index == 40:
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 400:
                continue
            rows = rows[::FRAME_STRIDE]
            windows.append(
                WindowSamples(
                    replicate=rep,
                    window_id=window_id,
                    lambda_value=float(rows[0]["lambda"]),
                    gap_kj_mol=_column(rows, "delta_e_shifted_kj_mol"),
                    e1_kj_mol=_column(rows, "E1_kj_mol"),
                    e2_kj_mol=_column(rows, "E2_kj_mol"),
                    evb_kj_mol=_column(rows, "Eevb_kj_mol"),
                )
            )
    return windows


def _column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _mbar_gap_pmf(windows: list[WindowSamples], bin_edges_kj_mol: np.ndarray | None = None) -> dict[str, np.ndarray]:
    beta = 1.0 / (R_KJ_MOL_K * TEMPERATURE_K)
    lambdas = np.asarray([window.lambda_value for window in windows], dtype=float)
    sample_counts = np.asarray([len(window.gap_kj_mol) for window in windows], dtype=float)
    gap = np.concatenate([window.gap_kj_mol for window in windows])
    e1 = np.concatenate([window.e1_kj_mol for window in windows])
    e2 = np.concatenate([window.e2_kj_mol for window in windows])
    evb = np.concatenate([window.evb_kj_mol for window in windows])

    mapped_energies = np.asarray(
        [(1.0 - lam) * e1 + lam * (e2 + DELTA_ALPHA_KJ_MOL) for lam in lambdas],
        dtype=float,
    )
    reduced = beta * mapped_energies
    target = beta * evb

    free_offsets = np.zeros(len(windows), dtype=float)
    for _ in range(MAX_MBAR_ITERATIONS):
        log_terms = np.log(sample_counts)[:, None] + free_offsets[:, None] - reduced
        log_denominator = _logsumexp(log_terms, axis=0)
        new_offsets = -_logsumexp(-reduced - log_denominator[None, :], axis=1)
        new_offsets -= new_offsets[0]
        if np.max(np.abs(new_offsets - free_offsets)) < 1.0e-10:
            free_offsets = new_offsets
            break
        free_offsets = new_offsets

    log_terms = np.log(sample_counts)[:, None] + free_offsets[:, None] - reduced
    log_denominator = _logsumexp(log_terms, axis=0)
    log_weights = -target - log_denominator
    log_weights -= _logsumexp(log_weights)
    weights = np.exp(log_weights)

    if bin_edges_kj_mol is None:
        lo = float(np.percentile(gap, 0.5))
        hi = float(np.percentile(gap, 99.5))
        bin_edges_kj_mol = np.linspace(lo, hi, 121)
    weighted_counts, edges = np.histogram(gap, bins=bin_edges_kj_mol, weights=weights)
    raw_counts, _ = np.histogram(gap, bins=bin_edges_kj_mol)
    probability = weighted_counts.astype(float)
    if probability.sum() > 0:
        probability /= probability.sum()
    free_energy = np.full_like(probability, np.nan, dtype=float)
    mask = probability > 0.0
    free_energy[mask] = -R_KJ_MOL_K * TEMPERATURE_K * np.log(probability[mask])
    if np.any(mask):
        free_energy[mask] -= np.nanmin(free_energy[mask])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "gap_kj_mol": centers,
        "gap_kcal_mol": centers * KJ_TO_KCAL,
        "free_energy_kj_mol": free_energy,
        "free_energy_kcal_mol": free_energy * KJ_TO_KCAL,
        "probability": probability,
        "raw_counts": raw_counts.astype(float),
        "bin_edges_kj_mol": edges,
    }


def _write_pmf_csv(path: Path, pmf: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kcal_mol", "free_energy_kcal_mol", "probability", "raw_counts"])
        for gap, free_energy, probability, raw_counts in zip(
            pmf["gap_kcal_mol"],
            pmf["free_energy_kcal_mol"],
            pmf["probability"],
            pmf["raw_counts"],
        ):
            writer.writerow([gap, "" if np.isnan(free_energy) else free_energy, probability, int(raw_counts)])


def _plot_pmf(path: Path, pmf: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    x = pmf["gap_kcal_mol"]
    y = pmf["free_energy_kcal_mol"]
    mask = np.isfinite(y)
    plt.figure(figsize=(7.0, 4.5), dpi=220)
    plt.plot(x[mask], y[mask], color="black", linewidth=2.0)
    _style_axes("EVB gap / kcal mol$^{-1}$", "Preliminary PMF / kcal mol$^{-1}$")
    plt.title("HG3.17 two-replica core mapped-window PMF")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_replicate_pmf(path: Path, combined: dict[str, np.ndarray], per_rep: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 4.5), dpi=220)
    for rep, pmf in sorted(per_rep.items()):
        y = pmf["free_energy_kcal_mol"]
        mask = np.isfinite(y)
        plt.plot(pmf["gap_kcal_mol"][mask], y[mask], linewidth=1.4, alpha=0.8, label=rep)
    y = combined["free_energy_kcal_mol"]
    mask = np.isfinite(y)
    plt.plot(combined["gap_kcal_mol"][mask], y[mask], color="black", linewidth=2.2, label="combined")
    _style_axes("EVB gap / kcal mol$^{-1}$", "Preliminary PMF / kcal mol$^{-1}$")
    plt.title("HG3.17 preliminary PMF by completed replicate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _write_summary(path: Path, windows: list[WindowSamples], combined: dict[str, np.ndarray], per_rep: dict[str, dict[str, np.ndarray]]) -> None:
    y = combined["free_energy_kcal_mol"]
    x = combined["gap_kcal_mol"]
    mask = np.isfinite(y)
    max_idx = int(np.nanargmax(y[mask]))
    finite_x = x[mask]
    finite_y = y[mask]
    summary = {
        "status": "preliminary_diagnostic_not_publication_quality",
        "replicates": sorted({window.replicate for window in windows}),
        "windows_used": "w001-w039 only; endpoint windows w000/w040 excluded because they show large excursions and poor overlap.",
        "n_windows": len(windows),
        "frames_used": int(sum(len(window.gap_kj_mol) for window in windows)),
        "frame_stride": FRAME_STRIDE,
        "mbar_iteration_cap": MAX_MBAR_ITERATIONS,
        "binning": {
            "gap_min_kcal_mol": float(combined["bin_edges_kj_mol"][0] * KJ_TO_KCAL),
            "gap_max_kcal_mol": float(combined["bin_edges_kj_mol"][-1] * KJ_TO_KCAL),
            "n_bins": int(len(combined["gap_kcal_mol"])),
        },
        "combined_profile": {
            "max_free_energy_kcal_mol": float(finite_y[max_idx]),
            "max_gap_kcal_mol": float(finite_x[max_idx]),
            "min_free_energy_kcal_mol": float(np.nanmin(finite_y)),
        },
        "limitations": [
            "Only two replicas are included, so uncertainty is not meaningful.",
            "The endpoint windows are excluded, so this is not a full reactant-to-product PMF.",
            "Gap overlap is still incomplete near the endpoints; do not report this as the EVB barrier.",
        ],
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _style_axes(xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        maximum = float(np.max(values))
        return maximum + np.log(np.sum(np.exp(values - maximum)))
    maxima = np.max(values, axis=axis, keepdims=True)
    shifted = values - maxima
    return np.squeeze(maxima, axis=axis) + np.log(np.sum(np.exp(shifted), axis=axis))


if __name__ == "__main__":
    main()
