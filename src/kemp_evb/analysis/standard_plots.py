from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EVBConfig

KJ_TO_KCAL = 1.0 / 4.184


def write_standard_analysis_plots(config: EVBConfig, output_dir: Path) -> dict[str, Any]:
    """Write standard, user-facing analysis plots in kcal/mol.

    The plots are deliberately diagnostic: they visualize PMF shape, window
    overlap, sampled gap tracking, EVB weights, and chemical coordinates. Barrier
    numbers are still governed by the explicit barrier-analysis machinery.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting stack
        return {"available": False, "warnings": [f"Could not import matplotlib for standard plots: {exc}"], "plots": []}

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    role_gaps = _load_role_gaps(config.output_dir)
    windows = _load_window_samples(Path(config.output_dir))
    pmf = _load_pmf(output_dir / "pmf_gap.csv")
    overlap = _load_overlap(output_dir / "window_overlap.json")

    plots: list[str] = []
    if pmf:
        path = plot_dir / "pmf_gap_kcal.png"
        _plot_pmf(plt, path, pmf, role_gaps)
        if path.exists():
            plots.append(str(path))
    if overlap is not None:
        path = plot_dir / "overlap_heatmap.png"
        _plot_overlap_heatmap(plt, path, overlap)
        if path.exists():
            plots.append(str(path))
        path = plot_dir / "adjacent_overlap.png"
        _plot_adjacent_overlap(plt, path, overlap)
        if path.exists():
            plots.append(str(path))
    if windows:
        path = plot_dir / "gap_tracking_and_weights.png"
        _plot_gap_tracking_and_weights(plt, path, windows)
        if path.exists():
            plots.append(str(path))
        path = plot_dir / "chemical_coordinates_by_window.png"
        _plot_chemical_coordinates(plt, path, windows)
        if path.exists():
            plots.append(str(path))
        path = plot_dir / "gap_vs_proton_transfer_scatter.png"
        _plot_gap_vs_reaction_coordinate(plt, path, windows, role_gaps)
        if path.exists():
            plots.append(str(path))

    summary = _plot_summary(windows, pmf, overlap, role_gaps)
    summary.update({"available": True, "plots": plots})
    (plot_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_pmf(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("free_energy_kj_mol"):
                continue
            rows.append(
                {
                    "gap_kcal_mol": float(row["gap_kj_mol"]) * KJ_TO_KCAL,
                    "free_energy_kcal_mol": float(row["free_energy_kj_mol"]) * KJ_TO_KCAL,
                    "counts": int(row["counts"]),
                }
            )
    return rows


def _load_overlap(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return None
    n = len(payload)
    matrix = np.zeros((n, n), dtype=float)
    for i, row in enumerate(payload):
        overlaps = row.get("overlaps", {})
        for j in range(n):
            matrix[i, j] = float(overlaps.get(f"u{j:03d}", overlaps.get(f"w{j:03d}", 0.0)))
    return matrix


def _load_window_samples(output_root: Path) -> list[dict[str, Any]]:
    windows_root = output_root / "windows"
    if not windows_root.exists():
        return []
    result = []
    for csv_path in sorted(windows_root.glob("*/production_observables.csv")):
        values: dict[str, list[float]] = {
            "gap": [],
            "donor_h": [],
            "h_acceptor": [],
            "proton_transfer_rc": [],
            "w2": [],
        }
        center = None
        window_id = csv_path.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if center is None and row.get("gap_center_kj_mol"):
                    center = float(row["gap_center_kj_mol"]) * KJ_TO_KCAL
                values["gap"].append(float(row["delta_e_shifted_kj_mol"]) * KJ_TO_KCAL)
                if row.get("donor_h"):
                    values["donor_h"].append(float(row["donor_h"]) * 10.0)
                if row.get("h_acceptor"):
                    values["h_acceptor"].append(float(row["h_acceptor"]) * 10.0)
                if row.get("proton_transfer_rc"):
                    values["proton_transfer_rc"].append(float(row["proton_transfer_rc"]) * 10.0)
                if row.get("w2"):
                    values["w2"].append(float(row["w2"]))
        if values["gap"]:
            result.append(
                {
                    "window_id": window_id,
                    "center_kcal_mol": center,
                    **{key: np.asarray(value, dtype=float) for key, value in values.items()},
                }
            )
    return result


def _load_role_gaps(output_root: str | Path) -> dict[str, float]:
    path = Path(output_root) / "analysis" / "evb_reference_fit_from_irc.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        role = payload["role_energies"]
        return {
            "E1": float(role["RC"]["gap_shifted_kj_mol"]) * KJ_TO_KCAL,
            "TS": float(role["TS"]["gap_shifted_kj_mol"]) * KJ_TO_KCAL,
            "E2": float(role["PROD"]["gap_shifted_kj_mol"]) * KJ_TO_KCAL,
        }
    except Exception:
        return {}
    return {}


def _plot_pmf(plt, path: Path, pmf: list[dict[str, float]], role_gaps: dict[str, float]) -> None:
    x = np.asarray([row["gap_kcal_mol"] for row in pmf])
    y = np.asarray([row["free_energy_kcal_mol"] for row in pmf])
    y = y - np.nanmin(y)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color="black", lw=2.0)
    _annotate_roles(ax, role_gaps)
    ax.set_xlabel("shifted EVB gap E1 - E2 - delta_alpha (kcal/mol)")
    ax.set_ylabel("free energy (kcal/mol)")
    ax.set_title("EVB gap PMF")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_overlap_heatmap(plt, path: Path, overlap: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(overlap, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
    ax.set_xlabel("window")
    ax.set_ylabel("window")
    ax.set_title("Window overlap matrix")
    fig.colorbar(image, ax=ax, label="histogram overlap")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_adjacent_overlap(plt, path: Path, overlap: np.ndarray) -> None:
    adjacent = np.asarray([overlap[i, i + 1] for i in range(overlap.shape[0] - 1)], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(len(adjacent)), adjacent, marker="o", ms=3, lw=1.3)
    ax.axhline(0.1, color="red", ls="--", lw=1.0, label="0.1 warning")
    ax.set_xlabel("neighboring window pair")
    ax.set_ylabel("overlap")
    ax.set_title("Neighboring-window overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gap_tracking_and_weights(plt, path: Path, windows: list[dict[str, Any]]) -> None:
    centers = np.asarray([w["center_kcal_mol"] for w in windows if w["center_kcal_mol"] is not None], dtype=float)
    if len(centers) == 0:
        return
    means = np.asarray([np.mean(w["gap"]) for w in windows if w["center_kcal_mol"] is not None], dtype=float)
    stds = np.asarray([np.std(w["gap"]) for w in windows if w["center_kcal_mol"] is not None], dtype=float)
    weights = np.asarray([np.mean(w["w2"]) if len(w["w2"]) else np.nan for w in windows if w["center_kcal_mol"] is not None])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].errorbar(centers, means, yerr=stds, fmt="o", ms=3, alpha=0.7, elinewidth=0.6)
    axes[0].plot([centers.min(), centers.max()], [centers.min(), centers.max()], color="black", lw=1)
    axes[0].set_xlabel("umbrella center (kcal/mol)")
    axes[0].set_ylabel("sampled gap mean +/- std (kcal/mol)")
    axes[0].set_title("Window target tracking")
    axes[1].plot(centers, weights, marker="o", ms=3, lw=1.2)
    axes[1].set_xlabel("umbrella center (kcal/mol)")
    axes[1].set_ylabel("mean state-2 EVB weight")
    axes[1].set_title("EVB mixing across windows")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_chemical_coordinates(plt, path: Path, windows: list[dict[str, Any]]) -> None:
    centers = np.asarray([w["center_kcal_mol"] for w in windows if w["center_kcal_mol"] is not None], dtype=float)
    if len(centers) == 0:
        return
    donor_h = np.asarray([np.mean(w["donor_h"]) if len(w["donor_h"]) else np.nan for w in windows if w["center_kcal_mol"] is not None])
    h_acceptor = np.asarray([np.mean(w["h_acceptor"]) if len(w["h_acceptor"]) else np.nan for w in windows if w["center_kcal_mol"] is not None])
    rc = np.asarray([np.mean(w["proton_transfer_rc"]) if len(w["proton_transfer_rc"]) else np.nan for w in windows if w["center_kcal_mol"] is not None])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(centers, donor_h, marker="o", ms=3, lw=1.2, label="donor-H")
    axes[0].plot(centers, h_acceptor, marker="s", ms=3, lw=1.2, ls="--", label="H-acceptor")
    axes[0].set_xlabel("umbrella center (kcal/mol)")
    axes[0].set_ylabel("mean distance (A)")
    axes[0].set_title("Chemical distances by window")
    axes[0].legend()
    axes[1].plot(centers, rc, marker="o", ms=3, lw=1.2)
    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].set_xlabel("umbrella center (kcal/mol)")
    axes[1].set_ylabel("d(donor-H) - d(H-acceptor) (A)")
    axes[1].set_title("Proton-transfer coordinate")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gap_vs_reaction_coordinate(plt, path: Path, windows: list[dict[str, Any]], role_gaps: dict[str, float]) -> None:
    gap = np.concatenate([w["gap"] for w in windows])
    rc_arrays = [w["proton_transfer_rc"] for w in windows if len(w["proton_transfer_rc"])]
    if not rc_arrays:
        return
    rc_values = np.concatenate(rc_arrays)
    if len(rc_values) != len(gap):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gap, rc_values, s=3, alpha=0.25)
    ax.axhline(0.0, color="black", lw=1.0)
    _annotate_roles(ax, role_gaps)
    ax.set_xlabel("sampled shifted gap (kcal/mol)")
    ax.set_ylabel("d(donor-H) - d(H-acceptor) (A)")
    ax.set_title("Chemical coordinate vs EVB gap")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _annotate_roles(ax, role_gaps: dict[str, float]) -> None:
    colors = {"E1": "#2ca02c", "TS": "#ff7f0e", "E2": "#9467bd"}
    for label, gap in role_gaps.items():
        ax.axvline(gap, color=colors.get(label, "gray"), ls="--", lw=1.1)
        ymin, ymax = ax.get_ylim()
        ax.text(gap, ymax, label, rotation=90, va="top", ha="right", color=colors.get(label, "gray"))


def _plot_summary(
    windows: list[dict[str, Any]], pmf: list[dict[str, float]], overlap: np.ndarray | None, role_gaps: dict[str, float]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"role_gaps_kcal_mol": role_gaps}
    if overlap is not None and overlap.shape[0] > 1:
        adjacent = np.asarray([overlap[i, i + 1] for i in range(overlap.shape[0] - 1)], dtype=float)
        summary["adjacent_overlap_min"] = float(adjacent.min())
        summary["adjacent_overlap_median"] = float(np.median(adjacent))
        summary["bad_adjacent_pairs_lt_0p1"] = [
            {"pair": f"u{i:03d}-u{i + 1:03d}", "overlap": float(value)}
            for i, value in enumerate(adjacent)
            if value < 0.1
        ]
    if pmf:
        summary["pmf_free_energy_range_kcal_mol"] = [
            float(min(row["free_energy_kcal_mol"] for row in pmf)),
            float(max(row["free_energy_kcal_mol"] for row in pmf)),
        ]
    if windows:
        gap = np.concatenate([window["gap"] for window in windows])
        summary["sampled_gap_range_kcal_mol"] = [float(np.min(gap)), float(np.max(gap))]
    return summary
