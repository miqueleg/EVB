from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DELTA_ALPHA_KJ_MOL = -53328.25448088075
KJ_TO_KCAL = 1.0 / 4.184
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "hg317_evb_mapping_replicates"
REPORT_DIR = OUTPUT_ROOT / "current_report"


@dataclass(slots=True)
class WindowProfile:
    replicate: str
    window_id: str
    lambda_value: float
    n_frames: int
    e1_mean: float
    e1_std: float
    e2_shifted_mean: float
    e2_shifted_std: float
    evb_mean: float
    evb_std: float
    gap_mean: float
    gap_std: float
    w1_mean: float
    w1_std: float
    w2_mean: float
    w2_std: float
    proton_rc_mean: float
    proton_rc_std: float
    ring_rc_mean: float
    ring_rc_std: float
    donor_h_mean: float
    h_acceptor_mean: float


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    completed_replicas = _completed_replicas()
    profiles = []
    for rep_dir in completed_replicas:
        profiles.extend(_load_replicate_profiles(rep_dir))
    if not profiles:
        raise SystemExit(f"No complete mapping windows found under {OUTPUT_ROOT}")

    _write_window_profile_csv(REPORT_DIR / "window_profiles.csv", profiles)
    aggregate = _aggregate_by_lambda(profiles)
    _write_aggregate_csv(REPORT_DIR / "aggregate_lambda_profile.csv", aggregate)
    _write_aggregate_kcal_csv(REPORT_DIR / "aggregate_lambda_profile_kcal.csv", aggregate)
    _write_summary(REPORT_DIR / "summary.json", completed_replicas, profiles, aggregate)
    _plot_energy_profile(REPORT_DIR / "energy_profile_lambda.png", aggregate)
    _plot_energy_profile(REPORT_DIR / "energy_profile_lambda_core.png", _core_rows(aggregate), renormalize=True)
    _plot_gap_profile(REPORT_DIR / "gap_profile_lambda.png", aggregate)
    _plot_gap_profile(REPORT_DIR / "gap_profile_lambda_core.png", _core_rows(aggregate))
    _plot_weights(REPORT_DIR / "evb_weights_lambda.png", aggregate)
    _plot_reaction_coordinates(REPORT_DIR / "reaction_coordinates_lambda.png", aggregate)
    _plot_gap_histograms(REPORT_DIR / "gap_histograms_by_lambda.png", completed_replicas)
    _plot_overlap_heatmaps(completed_replicas)
    _plot_adjacent_overlap(REPORT_DIR / "adjacent_gap_overlap.png", completed_replicas)
    _plot_existing_pmf(REPORT_DIR / "diagnostic_gap_pmf_replicates.png")
    print(f"Wrote HG3.17 mapping report for {len(completed_replicas)} complete replicas to {REPORT_DIR}")


def _completed_replicas() -> list[Path]:
    replicas = []
    for rep_dir in sorted(path for path in OUTPUT_ROOT.glob("rep*") if path.is_dir()):
        summary_files = sorted((rep_dir / "windows").glob("w*/summary.json"))
        if len(summary_files) == 41 and (rep_dir / "analysis" / "pmf_gap.csv").is_file():
            replicas.append(rep_dir)
    return replicas


def _load_replicate_profiles(rep_dir: Path) -> list[WindowProfile]:
    rows = []
    for csv_path in sorted((rep_dir / "windows").glob("w*/production_observables.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        if not data:
            continue
        window_id = data[0]["window_id"]
        lambda_value = float(data[0]["lambda"])
        e1 = _column(data, "E1_kj_mol")
        e2_shifted = _column(data, "E2_kj_mol") + DELTA_ALPHA_KJ_MOL
        evb = _column(data, "Eevb_kj_mol")
        gap = _column(data, "delta_e_shifted_kj_mol")
        w1 = _column(data, "w1")
        w2 = _column(data, "w2")
        proton_rc = _column(data, "proton_transfer_rc")
        ring_rc = _column(data, "ring_opening_rc")
        donor_h = _column(data, "donor_h")
        h_acceptor = _column(data, "h_acceptor")
        rows.append(
            WindowProfile(
                replicate=rep_dir.name,
                window_id=window_id,
                lambda_value=lambda_value,
                n_frames=len(data),
                e1_mean=_mean(e1),
                e1_std=_std(e1),
                e2_shifted_mean=_mean(e2_shifted),
                e2_shifted_std=_std(e2_shifted),
                evb_mean=_mean(evb),
                evb_std=_std(evb),
                gap_mean=_mean(gap),
                gap_std=_std(gap),
                w1_mean=_mean(w1),
                w1_std=_std(w1),
                w2_mean=_mean(w2),
                w2_std=_std(w2),
                proton_rc_mean=_mean(proton_rc),
                proton_rc_std=_std(proton_rc),
                ring_rc_mean=_mean(ring_rc),
                ring_rc_std=_std(ring_rc),
                donor_h_mean=_mean(donor_h),
                h_acceptor_mean=_mean(h_acceptor),
            )
        )
    return rows


def _column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _write_window_profile_csv(path: Path, profiles: list[WindowProfile]) -> None:
    fields = list(WindowProfile.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            writer.writerow({field: getattr(profile, field) for field in fields})


def _aggregate_by_lambda(profiles: list[WindowProfile]) -> list[dict[str, float]]:
    grouped: dict[float, list[WindowProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.lambda_value].append(profile)
    rows = []
    metrics = [
        "e1_mean",
        "e2_shifted_mean",
        "evb_mean",
        "gap_mean",
        "w1_mean",
        "w2_mean",
        "proton_rc_mean",
        "ring_rc_mean",
        "donor_h_mean",
        "h_acceptor_mean",
    ]
    for lambda_value in sorted(grouped):
        items = grouped[lambda_value]
        row: dict[str, float] = {
            "lambda": lambda_value,
            "n_replicates": float(len(items)),
            "n_frames_total": float(sum(item.n_frames for item in items)),
        }
        for metric in metrics:
            values = np.asarray([getattr(item, metric) for item in items], dtype=float)
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std_replicates"] = _std(values)
        rows.append(row)

    finite_evb = np.asarray([row["evb_mean_mean"] for row in rows], dtype=float)
    zero = float(np.min(finite_evb))
    for row in rows:
        row["evb_relative_kj_mol"] = row["evb_mean_mean"] - zero
        row["e1_relative_kj_mol"] = row["e1_mean_mean"] - zero
        row["e2_shifted_relative_kj_mol"] = row["e2_shifted_mean_mean"] - zero
    return rows


def _write_aggregate_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_kcal_csv(path: Path, rows: list[dict[str, float]]) -> None:
    energy_keys = {
        "e1_mean_mean",
        "e1_mean_std_replicates",
        "e2_shifted_mean_mean",
        "e2_shifted_mean_std_replicates",
        "evb_mean_mean",
        "evb_mean_std_replicates",
        "gap_mean_mean",
        "gap_mean_std_replicates",
        "evb_relative_kj_mol",
        "e1_relative_kj_mol",
        "e2_shifted_relative_kj_mol",
    }
    renamed = {
        "e1_mean_mean": "e1_mean_kcal_mol",
        "e1_mean_std_replicates": "e1_std_replicates_kcal_mol",
        "e2_shifted_mean_mean": "e2_shifted_mean_kcal_mol",
        "e2_shifted_mean_std_replicates": "e2_shifted_std_replicates_kcal_mol",
        "evb_mean_mean": "evb_mean_kcal_mol",
        "evb_mean_std_replicates": "evb_std_replicates_kcal_mol",
        "gap_mean_mean": "gap_mean_kcal_mol",
        "gap_mean_std_replicates": "gap_std_replicates_kcal_mol",
        "evb_relative_kj_mol": "evb_relative_kcal_mol",
        "e1_relative_kj_mol": "e1_relative_kcal_mol",
        "e2_shifted_relative_kj_mol": "e2_shifted_relative_kcal_mol",
    }
    converted_rows = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            output_key = renamed.get(key, key)
            converted[output_key] = value * KJ_TO_KCAL if key in energy_keys else value
        converted_rows.append(converted)
    fields = list(converted_rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(converted_rows)


def _write_summary(path: Path, replicas: list[Path], profiles: list[WindowProfile], aggregate: list[dict[str, float]]) -> None:
    evb = np.asarray([row["evb_relative_kj_mol"] for row in aggregate], dtype=float)
    lambdas = np.asarray([row["lambda"] for row in aggregate], dtype=float)
    gap = np.asarray([row["gap_mean_mean"] for row in aggregate], dtype=float)
    ts_idx = int(np.argmax(evb))
    summary = {
        "complete_replicates": [rep.name for rep in replicas],
        "n_complete_replicates": len(replicas),
        "n_complete_windows_per_replicate": 41,
        "frames_per_window": sorted({profile.n_frames for profile in profiles}),
        "total_frames_used": int(sum(profile.n_frames for profile in profiles)),
        "total_sampling_ps": float(sum(profile.n_frames for profile in profiles) * 0.5),
        "note": (
            "These are mapped-window diagnostic profiles from completed replicas only. "
            "The gap-PMF estimate is not yet a trustworthy barrier because early replicas show poor "
            "gap overlap near w000/w001 and the first mapping window contains large gap excursions."
        ),
        "diagnostic_evb_profile": {
            "minimum_relative_kj_mol": float(np.nanmin(evb)),
            "minimum_relative_kcal_mol": float(np.nanmin(evb) * KJ_TO_KCAL),
            "maximum_relative_kj_mol": float(np.nanmax(evb)),
            "maximum_relative_kcal_mol": float(np.nanmax(evb) * KJ_TO_KCAL),
            "max_lambda": float(lambdas[ts_idx]),
            "max_gap_kj_mol": float(gap[ts_idx]),
            "max_gap_kcal_mol": float(gap[ts_idx] * KJ_TO_KCAL),
            "profile_span_kj_mol": float(np.nanmax(evb) - np.nanmin(evb)),
            "profile_span_kcal_mol": float((np.nanmax(evb) - np.nanmin(evb)) * KJ_TO_KCAL),
        },
        "core_lambda_0p025_to_0p975": _core_profile_summary(aggregate),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _core_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    return [row for row in rows if 0.0 < row["lambda"] < 1.0]


def _core_profile_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    core = _core_rows(rows)
    evb_abs = np.asarray([row["evb_mean_mean"] for row in core], dtype=float)
    gap = np.asarray([row["gap_mean_mean"] for row in core], dtype=float)
    lambdas = np.asarray([row["lambda"] for row in core], dtype=float)
    relative = evb_abs - float(np.min(evb_abs))
    max_idx = int(np.argmax(relative))
    return {
        "n_windows": len(core),
        "minimum_relative_kj_mol": float(np.nanmin(relative)),
        "minimum_relative_kcal_mol": float(np.nanmin(relative) * KJ_TO_KCAL),
        "maximum_relative_kj_mol": float(np.nanmax(relative)),
        "maximum_relative_kcal_mol": float(np.nanmax(relative) * KJ_TO_KCAL),
        "max_lambda": float(lambdas[max_idx]),
        "max_gap_kj_mol": float(gap[max_idx]),
        "max_gap_kcal_mol": float(gap[max_idx] * KJ_TO_KCAL),
        "profile_span_kj_mol": float(np.nanmax(relative) - np.nanmin(relative)),
        "profile_span_kcal_mol": float((np.nanmax(relative) - np.nanmin(relative)) * KJ_TO_KCAL),
    }


def _plot_energy_profile(path: Path, rows: list[dict[str, float]], renormalize: bool = False) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray([row["lambda"] for row in rows], dtype=float)
    if renormalize:
        evb_abs = np.asarray([row["evb_mean_mean"] for row in rows], dtype=float)
        zero = float(np.min(evb_abs))
        e1 = np.asarray([row["e1_mean_mean"] - zero for row in rows], dtype=float)
        e2 = np.asarray([row["e2_shifted_mean_mean"] - zero for row in rows], dtype=float)
        evb = evb_abs - zero
    else:
        e1 = np.asarray([row["e1_relative_kj_mol"] for row in rows], dtype=float)
        e2 = np.asarray([row["e2_shifted_relative_kj_mol"] for row in rows], dtype=float)
        evb = np.asarray([row["evb_relative_kj_mol"] for row in rows], dtype=float)
    e1 *= KJ_TO_KCAL
    e2 *= KJ_TO_KCAL
    evb *= KJ_TO_KCAL
    plt.figure(figsize=(7.0, 4.6), dpi=220)
    plt.plot(x, e1, label="E1", color="#1f77b4", linewidth=1.8)
    plt.plot(x, e2, label="E2 + delta_alpha", color="#d62728", linewidth=1.8)
    plt.plot(x, evb, label="EVB lower surface", color="black", linewidth=2.1)
    _style_axes("Mapping lambda", "Relative mean potential energy / kcal mol$^{-1}$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_gap_profile(path: Path, rows: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray([row["lambda"] for row in rows], dtype=float)
    y = np.asarray([row["gap_mean_mean"] for row in rows], dtype=float)
    err = np.asarray([row["gap_mean_std_replicates"] for row in rows], dtype=float)
    y *= KJ_TO_KCAL
    err *= KJ_TO_KCAL
    plt.figure(figsize=(7.0, 4.4), dpi=220)
    plt.plot(x, y, color="#333333", linewidth=1.9)
    plt.fill_between(x, y - err, y + err, color="#888888", alpha=0.28, linewidth=0.0)
    _style_axes("Mapping lambda", "Mean EVB gap / kcal mol$^{-1}$")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_weights(path: Path, rows: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray([row["lambda"] for row in rows], dtype=float)
    plt.figure(figsize=(7.0, 4.4), dpi=220)
    plt.plot(x, [row["w1_mean_mean"] for row in rows], label="w1", color="#1f77b4", linewidth=1.9)
    plt.plot(x, [row["w2_mean_mean"] for row in rows], label="w2", color="#d62728", linewidth=1.9)
    plt.ylim(-0.03, 1.03)
    _style_axes("Mapping lambda", "EVB diabatic weight")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_reaction_coordinates(path: Path, rows: list[dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray([row["lambda"] for row in rows], dtype=float)
    plt.figure(figsize=(7.0, 4.4), dpi=220)
    plt.plot(x, [row["proton_rc_mean_mean"] for row in rows], label="proton-transfer RC", color="#2ca02c", linewidth=1.9)
    plt.plot(x, [row["ring_rc_mean_mean"] for row in rows], label="ring-opening RC", color="#9467bd", linewidth=1.9)
    _style_axes("Mapping lambda", "Reaction coordinate / nm")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_gap_histograms(path: Path, replicas: list[Path]) -> None:
    import matplotlib.pyplot as plt

    selected = {"w000", "w001", "w005", "w010", "w020", "w030", "w040"}
    values_by_window: dict[str, list[float]] = defaultdict(list)
    for rep in replicas:
        for csv_path in sorted((rep / "windows").glob("w*/production_observables.csv")):
            window_id = csv_path.parent.name
            if window_id not in selected:
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    values_by_window[window_id].append(float(row["delta_e_shifted_kj_mol"]))
    plt.figure(figsize=(7.2, 4.8), dpi=220)
    for window_id in sorted(values_by_window):
        values = np.asarray(values_by_window[window_id], dtype=float)
        if window_id == "w000":
            values = values[(values > np.percentile(values, 2.5)) & (values < np.percentile(values, 97.5))]
        values = values * KJ_TO_KCAL
        plt.hist(values, bins=40, density=True, histtype="step", linewidth=1.4, label=window_id)
    _style_axes("EVB gap / kcal mol$^{-1}$", "Density")
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_overlap_heatmaps(replicas: list[Path]) -> None:
    import matplotlib.pyplot as plt

    for rep in replicas:
        overlap_path = rep / "analysis" / "window_overlap.json"
        if not overlap_path.is_file():
            continue
        rows = json.loads(overlap_path.read_text(encoding="utf-8"))
        labels = [row["window_id"] for row in rows]
        matrix = np.asarray([[row["overlaps"][label] for label in labels] for row in rows], dtype=float)
        plt.figure(figsize=(6.4, 5.4), dpi=220)
        im = plt.imshow(matrix, origin="lower", cmap="viridis", vmin=0.0, vmax=max(0.25, float(np.nanmax(matrix))))
        ticks = np.arange(0, len(labels), 5)
        plt.xticks(ticks, [labels[i] for i in ticks], rotation=45, ha="right")
        plt.yticks(ticks, [labels[i] for i in ticks])
        plt.xlabel("Window")
        plt.ylabel("Window")
        plt.title(f"{rep.name} gap histogram overlap")
        plt.colorbar(im, label="overlap fraction")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"{rep.name}_overlap_heatmap.png", bbox_inches="tight")
        plt.close()


def _plot_adjacent_overlap(path: Path, replicas: list[Path]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 4.4), dpi=220)
    for rep in replicas:
        overlap_path = rep / "analysis" / "window_overlap.json"
        rows = json.loads(overlap_path.read_text(encoding="utf-8"))
        labels = [row["window_id"] for row in rows]
        adjacent = [rows[i]["overlaps"][labels[i + 1]] for i in range(len(labels) - 1)]
        x = np.arange(len(adjacent), dtype=float)
        plt.plot(x, adjacent, linewidth=1.6, marker="o", markersize=2.5, label=rep.name)
    _style_axes("Adjacent window pair index", "Gap histogram overlap")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_existing_pmf(path: Path) -> None:
    import matplotlib.pyplot as plt

    pmf_path = OUTPUT_ROOT / "summary" / "pmf_gap_replicates.csv"
    if not pmf_path.is_file():
        return
    gap = []
    mean = []
    std = []
    with pmf_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["mean_free_energy_kj_mol"]:
                continue
            gap.append(float(row["gap_kj_mol"]) * KJ_TO_KCAL)
            mean.append(float(row["mean_free_energy_kj_mol"]) * KJ_TO_KCAL)
            std.append(float(row["std_free_energy_kj_mol"]) * KJ_TO_KCAL if row["std_free_energy_kj_mol"] else math.nan)
    if not gap:
        return
    x = np.asarray(gap, dtype=float)
    y = np.asarray(mean, dtype=float)
    yerr = np.asarray(std, dtype=float)
    plt.figure(figsize=(7.0, 4.4), dpi=220)
    plt.plot(x, y, color="black", linewidth=1.8)
    if np.any(np.isfinite(yerr)):
        plt.fill_between(x, y - yerr, y + yerr, color="#999999", alpha=0.25, linewidth=0.0)
    _style_axes("EVB gap / kcal mol$^{-1}$", "Diagnostic PMF / kcal mol$^{-1}$")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _style_axes(xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


if __name__ == "__main__":
    main()
