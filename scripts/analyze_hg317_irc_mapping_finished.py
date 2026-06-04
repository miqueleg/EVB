from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "hg317_irc_mapping_barrier_calibrated_replicates"
SUMMARY = OUTPUT_ROOT / "summary_finished"
KJ_TO_KCAL = 1.0 / 4.184
N_WINDOWS = 41
N_REPLICATES_TOTAL = 30
PS_PER_WINDOW = 205.0


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    complete = _complete_replicates()
    if not complete:
        raise SystemExit(f"No complete analyzed replicas found under {OUTPUT_ROOT}")

    pmfs = [_load_pmf(rep / "analysis" / "pmf_gap.csv") for rep in complete]
    barriers = [_load_json(rep / "analysis" / "barrier_estimate.json") for rep in complete]
    stats = [_load_json(rep / "analysis" / "window_statistics.json") for rep in complete]
    overlaps = [_load_json(rep / "analysis" / "window_overlap.json") for rep in complete]
    progress = _progress()

    _write_barrier_table(complete, barriers)
    _write_mean_pmf(complete, pmfs)
    _write_window_gap_table(complete, stats)
    _write_overlap_table(complete, overlaps)
    _plot_pmfs(complete, pmfs)
    _plot_count_filtered_gap_pmf(pmfs)
    _plot_proton_transfer_rc_diagnostic(complete)
    chemical_rc_metrics = _plot_proton_transfer_rc_replicates(complete)
    _plot_chemical_progress_by_lambda(complete)
    _plot_barriers(complete, barriers)
    _plot_mean_gap(complete, stats)
    _plot_adjacent_overlap(complete, overlaps)
    _plot_mean_overlap_heatmap(complete, overlaps)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "complete_replicates": [rep.name for rep in complete],
        "n_complete_replicates": len(complete),
        "progress": progress,
        "barrier_summary_kcal_mol": _barrier_summary(barriers),
        "proton_transfer_rc_summary_kcal_mol": chemical_rc_metrics,
        "overlap_summary": _overlap_summary(overlaps),
        "warning": (
            "These are interim diagnostics from completed replicas only. Several adjacent windows have weak overlap, "
            "so the PMF should not be treated as converged until overlap improves and the full replica set is analyzed."
        ),
    }
    (SUMMARY / "finished_replicates_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload)
    print(json.dumps(payload, indent=2))


def _complete_replicates() -> list[Path]:
    reps = []
    for rep in sorted(OUTPUT_ROOT.glob("rep*")):
        if not rep.is_dir():
            continue
        if not (rep / "series" / "window_index.json").is_file():
            continue
        if not (rep / "analysis" / "pmf_gap.csv").is_file():
            continue
        summaries = list((rep / "windows").glob("w*/summary.json"))
        if len(summaries) == N_WINDOWS:
            reps.append(rep)
    return reps


def _progress() -> dict:
    rows = []
    for rep in sorted(OUTPUT_ROOT.glob("rep*")):
        windows = rep / "windows"
        if not windows.is_dir():
            continue
        completed = sorted(windows.glob("w*/summary.json"))
        active = sorted(p.parent.name for p in windows.glob("w*/production_observables.csv") if not (p.parent / "summary.json").is_file())
        last_time = max((p.stat().st_mtime for p in completed), default=None)
        rows.append(
            {
                "replica": rep.name,
                "completed_windows": len(completed),
                "active_windows": active,
                "last_completed_window": completed[-1].parent.name if completed else None,
                "last_completed_at": None if last_time is None else datetime.fromtimestamp(last_time).isoformat(timespec="seconds"),
            }
        )
    completed_windows = sum(row["completed_windows"] for row in rows)
    total_windows = N_REPLICATES_TOTAL * N_WINDOWS
    return {
        "replicas": rows,
        "completed_windows": completed_windows,
        "total_windows": total_windows,
        "completed_fraction": completed_windows / total_windows if total_windows else 0.0,
        "completed_ns": completed_windows * PS_PER_WINDOW * 1.0e-3,
        "total_ns": total_windows * PS_PER_WINDOW * 1.0e-3,
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pmf(path: Path) -> dict[str, np.ndarray]:
    gap = []
    free = []
    counts = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gap.append(float(row["gap_kj_mol"]) * KJ_TO_KCAL)
            free.append(float(row["free_energy_kj_mol"]) * KJ_TO_KCAL if row["free_energy_kj_mol"] else np.nan)
            counts.append(int(row["counts"]))
    return {"gap_kcal_mol": np.asarray(gap), "free_kcal_mol": np.asarray(free), "counts": np.asarray(counts)}


def _write_barrier_table(reps: list[Path], barriers: list[dict]) -> None:
    with (SUMMARY / "barriers_kcal_mol.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replica", "barrier_kcal_mol", "reaction_free_energy_kcal_mol", "ts_gap_kcal_mol", "reactant_gap_kcal_mol"])
        for rep, barrier in zip(reps, barriers):
            writer.writerow(
                [
                    rep.name,
                    _kcal_or_blank(barrier.get("barrier_forward_kj_mol")),
                    _kcal_or_blank(barrier.get("reaction_free_energy_kj_mol")),
                    _kcal_or_blank(barrier.get("ts_gap_kj_mol")),
                    _kcal_or_blank(barrier.get("reactant_gap_kj_mol")),
                ]
            )


def _write_mean_pmf(reps: list[Path], pmfs: list[dict[str, np.ndarray]]) -> None:
    gap = pmfs[0]["gap_kcal_mol"]
    matrix = np.asarray([pmf["free_kcal_mol"] for pmf in pmfs], dtype=float)
    counts = np.asarray([pmf["counts"] for pmf in pmfs], dtype=int)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1) if len(pmfs) > 1 else np.zeros_like(mean)
    valid = np.isfinite(mean)
    if np.any(valid):
        mean[valid] -= np.nanmin(mean[valid])
    with (SUMMARY / "pmf_gap_mean_kcal_mol.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kcal_mol", "mean_free_energy_kcal_mol", "std_free_energy_kcal_mol", "total_counts"])
        for x, y, err, count in zip(gap, mean, std, counts.sum(axis=0)):
            writer.writerow([x, _finite_or_blank(y), _finite_or_blank(err), int(count)])


def _write_window_gap_table(reps: list[Path], stats: list[list[dict]]) -> None:
    with (SUMMARY / "window_gap_statistics_kcal_mol.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replica", "window_id", "lambda", "mean_gap_kcal_mol", "std_gap_kcal_mol", "effective_sample_size"])
        for rep, table in zip(reps, stats):
            for row in table:
                window_id = row["window_id"]
                lam = int(window_id[1:]) / (N_WINDOWS - 1)
                writer.writerow(
                    [
                        rep.name,
                        window_id,
                        lam,
                        row["mean_gap_kj_mol"] * KJ_TO_KCAL,
                        row["std_gap_kj_mol"] * KJ_TO_KCAL,
                        row["effective_sample_size"],
                    ]
                )


def _write_overlap_table(reps: list[Path], overlaps: list[list[dict]]) -> None:
    with (SUMMARY / "adjacent_overlap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replica", "left_window", "right_window", "overlap"])
        for rep, table in zip(reps, overlaps):
            lookup = {row["window_id"]: row["overlaps"] for row in table}
            for index in range(N_WINDOWS - 1):
                left = f"w{index:03d}"
                right = f"w{index + 1:03d}"
                writer.writerow([rep.name, left, right, lookup.get(left, {}).get(right, np.nan)])


def _plot_pmfs(reps: list[Path], pmfs: list[dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap = pmfs[0]["gap_kcal_mol"]
    matrix = np.asarray([pmf["free_kcal_mol"] for pmf in pmfs], dtype=float)
    counts = np.asarray([pmf["counts"] for pmf in pmfs], dtype=int).sum(axis=0)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1) if len(pmfs) > 1 else np.zeros_like(mean)
    valid = np.isfinite(mean) & (counts > 0)
    if np.any(valid):
        mean[valid] -= np.nanmin(mean[valid])
    focused = valid & (mean < 250.0)

    plt.figure(figsize=(7.2, 4.6), dpi=220)
    for rep, pmf in zip(reps, pmfs):
        y = pmf["free_kcal_mol"].copy()
        mask = np.isfinite(y)
        if np.any(mask):
            y[mask] -= np.nanmin(y[mask])
        plt.plot(gap[focused], y[focused], color="0.75", linewidth=0.9, alpha=0.75)
    plt.plot(gap[focused], mean[focused], color="black", linewidth=2.0, label=f"mean, n={len(reps)}")
    plt.fill_between(gap[focused], mean[focused] - std[focused], mean[focused] + std[focused], color="0.2", alpha=0.18, linewidth=0)
    plt.xlabel("EVB gap (kcal/mol)")
    plt.ylabel("Free energy (kcal/mol)")
    plt.title("HG3.17 barrier-calibrated mapping PMF, completed replicas")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "pmf_gap_finished_replicates_kcal_mol.png")
    plt.close()


def _plot_count_filtered_gap_pmf(pmfs: list[dict[str, np.ndarray]], min_counts: int = 1000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap = pmfs[0]["gap_kcal_mol"]
    matrix = np.asarray([pmf["free_kcal_mol"] for pmf in pmfs], dtype=float)
    counts = np.asarray([pmf["counts"] for pmf in pmfs], dtype=int).sum(axis=0)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1) if len(pmfs) > 1 else np.zeros_like(mean)
    valid = np.isfinite(mean) & (counts >= min_counts)
    if np.any(valid):
        mean[valid] -= np.nanmin(mean[valid])

    with (SUMMARY / "pmf_gap_count_filtered_kcal_mol.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gap_kcal_mol", "mean_free_energy_kcal_mol", "std_free_energy_kcal_mol", "total_counts"])
        for x, y, err, count, keep in zip(gap, mean, std, counts, valid):
            if keep:
                writer.writerow([x, y, err, int(count)])

    plt.figure(figsize=(7.2, 4.6), dpi=220)
    plt.plot(gap[valid], mean[valid], color="black", linewidth=2.0, label=f"counts >= {min_counts}")
    plt.fill_between(gap[valid], mean[valid] - std[valid], mean[valid] + std[valid], color="0.2", alpha=0.18, linewidth=0)
    plt.xlabel("EVB gap (kcal/mol)")
    plt.ylabel("Free energy (kcal/mol)")
    plt.title("Count-filtered EVB gap PMF")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "pmf_gap_count_filtered_kcal_mol.png")
    plt.close()


def _plot_proton_transfer_rc_diagnostic(reps: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = []
    for rep in reps:
        for path in sorted((rep / "windows").glob("w*/production_observables.csv")):
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    value = row.get("proton_transfer_rc_nm") or row.get("proton_transfer_rc")
                    if value:
                        values.append(float(value) * 10.0)
    if not values:
        return
    values_array = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values_array, [0.5, 99.5])
    counts, edges = np.histogram(values_array, bins=160, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    probability = counts.astype(float) / counts.sum()
    free_energy = np.full_like(probability, np.nan, dtype=float)
    mask = probability > 0.0
    rt_kcal = 0.00198720425864083 * 300.0
    free_energy[mask] = -rt_kcal * np.log(probability[mask])
    free_energy[mask] -= np.nanmin(free_energy[mask])

    with (SUMMARY / "proton_transfer_rc_diagnostic_pmf.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["proton_transfer_rc_angstrom", "free_energy_kcal_mol", "counts"])
        for center, fe, count in zip(centers, free_energy, counts):
            writer.writerow([center, _finite_or_blank(fe), int(count)])

    plt.figure(figsize=(7.2, 4.6), dpi=220)
    plt.plot(centers[mask], free_energy[mask], color="black", linewidth=2.0)
    plt.xlabel("d(donor-H) - d(H-acceptor) (A)")
    plt.ylabel("Diagnostic free energy (kcal/mol)")
    plt.title("Pooled proton-transfer coordinate diagnostic")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "proton_transfer_rc_diagnostic_pmf.png")
    plt.close()


def _plot_proton_transfer_rc_replicates(reps: list[Path]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values_by_rep = []
    pooled = []
    for rep in reps:
        values = []
        for path in sorted((rep / "windows").glob("w*/production_observables.csv")):
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    value = row.get("proton_transfer_rc_nm") or row.get("proton_transfer_rc")
                    if value:
                        values.append(float(value) * 10.0)
        if values:
            arr = np.asarray(values, dtype=float)
            values_by_rep.append((rep.name, arr))
            pooled.append(arr)
    if not pooled:
        return {}

    pooled_values = np.concatenate(pooled)
    lo, hi = np.nanpercentile(pooled_values, [0.5, 99.5])
    bins = np.linspace(lo, hi, 151)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rt_kcal = 0.00198720425864083 * 300.0

    curves = []
    metrics = []
    for rep_name, values in values_by_rep:
        counts, _ = np.histogram(values, bins=bins)
        probability = counts.astype(float) / counts.sum()
        free_energy = np.full_like(probability, np.nan, dtype=float)
        mask = probability > 0.0
        free_energy[mask] = -rt_kcal * np.log(probability[mask])
        reactant_mask = mask & (centers < -0.5)
        zero = np.nanmin(free_energy[reactant_mask]) if np.any(reactant_mask) else np.nanmin(free_energy[mask])
        free_energy[mask] -= zero
        curves.append(free_energy)
        metrics.append(_chemical_rc_barrier_metrics(rep_name, centers, free_energy))

    matrix = np.asarray(curves, dtype=float)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    mean_mask = np.isfinite(mean)

    with (SUMMARY / "proton_transfer_rc_replicate_pmf.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["proton_transfer_rc_angstrom", "mean_free_energy_kcal_mol", "std_free_energy_kcal_mol"])
        for center, fe, err in zip(centers, mean, std):
            writer.writerow([center, _finite_or_blank(fe), _finite_or_blank(err)])

    with (SUMMARY / "proton_transfer_rc_barriers_kcal_mol.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["replica", "reactant_min_kcal_mol", "ts_max_kcal_mol", "product_min_kcal_mol", "barrier_kcal_mol"])
        writer.writeheader()
        writer.writerows(metrics)

    plt.figure(figsize=(7.4, 4.8), dpi=220)
    for rep_name, curve in zip([name for name, _ in values_by_rep], curves):
        mask = np.isfinite(curve)
        plt.plot(centers[mask], curve[mask], color="0.72", linewidth=0.9, alpha=0.8)
    plt.plot(centers[mean_mask], mean[mean_mask], color="black", linewidth=2.2, label=f"mean, n={len(curves)}")
    plt.fill_between(
        centers[mean_mask],
        mean[mean_mask] - std[mean_mask],
        mean[mean_mask] + std[mean_mask],
        color="0.2",
        alpha=0.16,
        linewidth=0,
    )
    reactant_mask = mean_mask & (centers < -0.5)
    ts_mask = mean_mask & (centers >= -0.5) & (centers <= 0.5)
    product_mask = mean_mask & (centers > 0.5)
    if np.any(reactant_mask) and np.any(ts_mask) and np.any(product_mask):
        reactant_indices = np.where(reactant_mask)[0]
        ts_indices = np.where(ts_mask)[0]
        product_indices = np.where(product_mask)[0]
        reactant_index = reactant_indices[int(np.nanargmin(mean[reactant_indices]))]
        ts_index = ts_indices[int(np.nanargmax(mean[ts_indices]))]
        product_index = product_indices[int(np.nanargmin(mean[product_indices]))]
        barrier = mean[ts_index] - mean[reactant_index]
        plt.scatter(
            [centers[reactant_index], centers[ts_index], centers[product_index]],
            [mean[reactant_index], mean[ts_index], mean[product_index]],
            color=["#2f6f4e", "#b34747", "#315f8f"],
            s=34,
            zorder=5,
        )
        plt.annotate(
            f"barrier {barrier:.2f} kcal/mol",
            xy=(centers[ts_index], mean[ts_index]),
            xytext=(centers[ts_index] + 0.35, mean[ts_index] + 0.55),
            arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "black"},
            fontsize=8.5,
        )
    plt.axvline(0.0, color="#b34747", linestyle="--", linewidth=1.0, label="H midway")
    plt.text(lo, np.nanmax(mean[mean_mask]) * 0.92, "RC", ha="left", va="top")
    plt.text(hi, np.nanmax(mean[mean_mask]) * 0.92, "PROD", ha="right", va="top")
    plt.xlabel("d(C-H donor) - d(H-O acceptor) (A)")
    plt.ylabel("Diagnostic free energy (kcal/mol)")
    plt.title("HG3.17 proton-transfer coordinate, completed replicas")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "proton_transfer_rc_replicates_kcal_mol.png")
    plt.close()

    barrier_values = [row["barrier_kcal_mol"] for row in metrics if row["barrier_kcal_mol"] != ""]
    product_values = [row["product_min_kcal_mol"] for row in metrics if row["product_min_kcal_mol"] != ""]
    return {
        "coordinate": "d(C-H donor) - d(H-O acceptor)",
        "n_replicates": len(metrics),
        "barrier_mean": float(np.mean(barrier_values)) if barrier_values else None,
        "barrier_std": float(np.std(barrier_values, ddof=1)) if len(barrier_values) > 1 else 0.0 if barrier_values else None,
        "product_min_mean": float(np.mean(product_values)) if product_values else None,
        "product_min_std": float(np.std(product_values, ddof=1)) if len(product_values) > 1 else 0.0 if product_values else None,
        "note": "Diagnostic pooled-coordinate PMF. Mapping windows were not biased directly along this geometrical coordinate.",
    }


def _chemical_rc_barrier_metrics(replica: str, centers: np.ndarray, free_energy: np.ndarray) -> dict:
    mask = np.isfinite(free_energy)
    reactant_mask = mask & (centers < -0.5)
    ts_mask = mask & (centers >= -0.5) & (centers <= 0.5)
    product_mask = mask & (centers > 0.5)
    reactant_min = float(np.nanmin(free_energy[reactant_mask])) if np.any(reactant_mask) else ""
    ts_max = float(np.nanmax(free_energy[ts_mask])) if np.any(ts_mask) else ""
    product_min = float(np.nanmin(free_energy[product_mask])) if np.any(product_mask) else ""
    barrier = "" if ts_max == "" or reactant_min == "" else ts_max - reactant_min
    return {
        "replica": replica,
        "reactant_min_kcal_mol": reactant_min,
        "ts_max_kcal_mol": ts_max,
        "product_min_kcal_mol": product_min,
        "barrier_kcal_mol": barrier,
    }


def _plot_chemical_progress_by_lambda(reps: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = np.asarray([index / (N_WINDOWS - 1) for index in range(N_WINDOWS)], dtype=float)
    rc_rows = []
    gap_rows = []
    for rep in reps:
        rc = []
        gap = []
        for index in range(N_WINDOWS):
            path = rep / "windows" / f"w{index:03d}" / "production_observables.csv"
            rc_values = []
            gap_values = []
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rc_values.append(float(row["proton_transfer_rc_nm"]) * 10.0)
                    gap_values.append(float(row["delta_e_shifted_kj_mol"]) * KJ_TO_KCAL)
            rc.append(float(np.mean(rc_values)))
            gap.append(float(np.mean(gap_values)))
        rc_rows.append(rc)
        gap_rows.append(gap)

    rc_matrix = np.asarray(rc_rows, dtype=float)
    gap_matrix = np.asarray(gap_rows, dtype=float)
    rc_mean = np.mean(rc_matrix, axis=0)
    rc_std = np.std(rc_matrix, axis=0, ddof=1) if len(reps) > 1 else np.zeros_like(rc_mean)
    gap_mean = np.mean(gap_matrix, axis=0)
    gap_std = np.std(gap_matrix, axis=0, ddof=1) if len(reps) > 1 else np.zeros_like(gap_mean)

    with (SUMMARY / "chemical_progress_by_lambda.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "mean_proton_transfer_rc_angstrom", "std_proton_transfer_rc_angstrom", "mean_gap_kcal_mol", "std_gap_kcal_mol"])
        for row in zip(lambdas, rc_mean, rc_std, gap_mean, gap_std):
            writer.writerow(row)

    plt.figure(figsize=(7.4, 4.8), dpi=220)
    for row in rc_matrix:
        plt.plot(lambdas, row, color="0.72", linewidth=0.9, alpha=0.8)
    plt.plot(lambdas, rc_mean, color="black", linewidth=2.2, label=f"mean, n={len(reps)}")
    plt.fill_between(lambdas, rc_mean - rc_std, rc_mean + rc_std, color="0.2", alpha=0.16, linewidth=0)
    plt.axhline(0.0, color="#b34747", linestyle="--", linewidth=1.0, label="H midway")
    plt.text(0.0, np.nanmax(rc_mean + rc_std) * 0.92, "RC", ha="left", va="top")
    plt.text(1.0, np.nanmax(rc_mean + rc_std) * 0.92, "PROD", ha="right", va="top")
    plt.xlabel("Mapping lambda")
    plt.ylabel("d(C-H donor) - d(H-O acceptor) (A)")
    plt.title("Chemical progress by mapping lambda")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "chemical_progress_by_lambda.png")
    plt.close()


def _plot_barriers(reps: list[Path], barriers: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [rep.name for rep in reps]
    values = np.asarray([barrier["barrier_forward_kj_mol"] * KJ_TO_KCAL for barrier in barriers], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    plt.figure(figsize=(7.0, 4.2), dpi=220)
    plt.bar(labels, values, color="#3f6f8f")
    plt.axhline(mean, color="black", linewidth=1.5, label=f"mean {mean:.2f} +/- {std:.2f}")
    plt.ylabel("Barrier (kcal/mol)")
    plt.title("Per-replica PMF barrier estimates")
    plt.xticks(rotation=35, ha="right")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(SUMMARY / "barriers_finished_replicates_kcal_mol.png")
    plt.close()


def _plot_mean_gap(reps: list[Path], stats: list[list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = np.asarray([index / (N_WINDOWS - 1) for index in range(N_WINDOWS)], dtype=float)
    matrix = []
    for table in stats:
        lookup = {row["window_id"]: row["mean_gap_kj_mol"] * KJ_TO_KCAL for row in table}
        matrix.append([lookup[f"w{index:03d}"] for index in range(N_WINDOWS)])
    matrix = np.asarray(matrix, dtype=float)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1) if len(stats) > 1 else np.zeros_like(mean)
    plt.figure(figsize=(7.0, 4.2), dpi=220)
    for row in matrix:
        plt.plot(lambdas, row, color="0.75", linewidth=0.9)
    plt.plot(lambdas, mean, color="black", linewidth=2.0)
    plt.fill_between(lambdas, mean - std, mean + std, color="0.2", alpha=0.18, linewidth=0)
    plt.xlabel("Mapping lambda")
    plt.ylabel("Mean EVB gap (kcal/mol)")
    plt.title("Window mean gap by mapping lambda")
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "mean_gap_by_lambda_finished_replicates.png")
    plt.close()


def _plot_adjacent_overlap(reps: list[Path], overlaps: list[list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(N_WINDOWS - 1)
    matrix = []
    for table in overlaps:
        lookup = {row["window_id"]: row["overlaps"] for row in table}
        matrix.append([lookup[f"w{index:03d}"].get(f"w{index + 1:03d}", np.nan) for index in range(N_WINDOWS - 1)])
    matrix = np.asarray(matrix, dtype=float)
    mean = np.nanmean(matrix, axis=0)
    plt.figure(figsize=(7.2, 4.2), dpi=220)
    for row in matrix:
        plt.plot(x, row, color="0.75", linewidth=0.9)
    plt.plot(x, mean, color="black", linewidth=2.0)
    plt.axhline(0.03, color="#b34747", linestyle="--", linewidth=1.2, label="weak-overlap guide")
    plt.xlabel("Adjacent window pair index")
    plt.ylabel("Histogram overlap")
    plt.title("Adjacent window overlap")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SUMMARY / "adjacent_overlap_finished_replicates.png")
    plt.close()


def _plot_mean_overlap_heatmap(reps: list[Path], overlaps: list[list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = []
    for table in overlaps:
        lookup = {row["window_id"]: row["overlaps"] for row in table}
        matrix = np.zeros((N_WINDOWS, N_WINDOWS), dtype=float)
        for i in range(N_WINDOWS):
            for j in range(N_WINDOWS):
                matrix[i, j] = lookup.get(f"w{i:03d}", {}).get(f"w{j:03d}", 0.0)
        matrices.append(matrix)
    mean = np.mean(np.asarray(matrices), axis=0)
    plt.figure(figsize=(6.2, 5.2), dpi=220)
    im = plt.imshow(mean, origin="lower", cmap="viridis", vmin=0.0, vmax=min(0.2, float(np.max(mean))))
    plt.xlabel("Window")
    plt.ylabel("Window")
    plt.title("Mean window-overlap matrix")
    plt.colorbar(im, label="overlap")
    plt.tight_layout()
    plt.savefig(SUMMARY / "mean_window_overlap_heatmap.png")
    plt.close()


def _barrier_summary(barriers: list[dict]) -> dict:
    barrier = np.asarray([row["barrier_forward_kj_mol"] * KJ_TO_KCAL for row in barriers], dtype=float)
    reaction = np.asarray([row["reaction_free_energy_kj_mol"] * KJ_TO_KCAL for row in barriers], dtype=float)
    return {
        "barrier_mean": float(np.mean(barrier)),
        "barrier_std": float(np.std(barrier, ddof=1)) if len(barrier) > 1 else 0.0,
        "reaction_free_energy_mean": float(np.mean(reaction)),
        "reaction_free_energy_std": float(np.std(reaction, ddof=1)) if len(reaction) > 1 else 0.0,
        "barrier_values": [float(value) for value in barrier],
        "reaction_free_energy_values": [float(value) for value in reaction],
    }


def _overlap_summary(overlaps: list[list[dict]]) -> dict:
    adjacent = []
    for table in overlaps:
        lookup = {row["window_id"]: row["overlaps"] for row in table}
        adjacent.extend(lookup[f"w{index:03d}"].get(f"w{index + 1:03d}", np.nan) for index in range(N_WINDOWS - 1))
    adjacent_array = np.asarray(adjacent, dtype=float)
    return {
        "adjacent_overlap_min": float(np.nanmin(adjacent_array)),
        "adjacent_overlap_mean": float(np.nanmean(adjacent_array)),
        "adjacent_overlap_fraction_below_0_03": float(np.mean(adjacent_array < 0.03)),
    }


def _write_markdown(payload: dict) -> None:
    barrier = payload["barrier_summary_kcal_mol"]
    overlap = payload["overlap_summary"]
    progress = payload["progress"]
    lines = [
        "# HG3.17 Finished-Replica Interim Analysis",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Complete replicas analyzed: `{payload['n_complete_replicates']}`",
        f"Completed windows: `{progress['completed_windows']}` / `{progress['total_windows']}`",
        f"Completed simulation: `{progress['completed_ns']:.3f}` / `{progress['total_ns']:.3f}` ns",
        "",
        "## PMF Barriers",
        "",
        f"- Barrier mean: `{barrier['barrier_mean']:.3f}` kcal/mol",
        f"- Barrier std: `{barrier['barrier_std']:.3f}` kcal/mol",
        f"- Reaction free energy mean: `{barrier['reaction_free_energy_mean']:.3f}` kcal/mol",
        f"- Reaction free energy std: `{barrier['reaction_free_energy_std']:.3f}` kcal/mol",
        "",
        "## Overlap",
        "",
        f"- Adjacent overlap minimum: `{overlap['adjacent_overlap_min']:.4f}`",
        f"- Adjacent overlap mean: `{overlap['adjacent_overlap_mean']:.4f}`",
        f"- Fraction of adjacent overlaps below 0.03: `{overlap['adjacent_overlap_fraction_below_0_03']:.3f}`",
        "",
        "## Warning",
        "",
        payload["warning"],
    ]
    (SUMMARY / "finished_replicates_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kcal_or_blank(value) -> float | str:
    return "" if value is None else float(value) * KJ_TO_KCAL


def _finite_or_blank(value) -> float | str:
    return "" if not np.isfinite(value) else float(value)


if __name__ == "__main__":
    main()
