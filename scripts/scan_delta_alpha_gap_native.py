from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kemp_evb.cli import run_sample_series  # noqa: E402
from kemp_evb.config import EVBConfig, ProjectSettings, load_config  # noqa: E402


def summarize_window_csv(csv_path: Path) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "delta_e_shifted_kj_mol": float(row["delta_e_shifted_kj_mol"]),
                    "w1": float(row["w1"]),
                    "w2": float(row["w2"]),
                    "proton_transfer_rc": float(row["proton_transfer_rc"]),
                    "ring_opening_rc": float(row["ring_opening_rc"]),
                }
            )
    best = min(rows, key=lambda row: abs(row["delta_e_shifted_kj_mol"]))
    mixed = min(rows, key=lambda row: abs(row["w2"] - 0.5))
    return {
        "n_frames": len(rows),
        "gap_min": min(row["delta_e_shifted_kj_mol"] for row in rows),
        "gap_max": max(row["delta_e_shifted_kj_mol"] for row in rows),
        "closest_gap_to_zero": best["delta_e_shifted_kj_mol"],
        "w2_at_closest_gap": best["w2"],
        "pt_at_closest_gap": best["proton_transfer_rc"],
        "ro_at_closest_gap": best["ring_opening_rc"],
        "closest_mixing_w2": mixed["w2"],
        "gap_at_closest_mixing": mixed["delta_e_shifted_kj_mol"],
        "pt_at_closest_mixing": mixed["proton_transfer_rc"],
        "ro_at_closest_mixing": mixed["ring_opening_rc"],
    }


def summarize_output(output_dir: Path) -> dict[str, Any]:
    windows_dir = output_dir / "windows"
    summaries: list[dict[str, Any]] = []
    for csv_path in sorted(windows_dir.glob("u*/production_observables.csv")):
        data = summarize_window_csv(csv_path)
        data["window_id"] = csv_path.parent.name
        summaries.append(data)
    if not summaries:
        raise ValueError(f"No gap-umbrella window CSV files found under {windows_dir}")
    best_gap = min(summaries, key=lambda row: abs(row["closest_gap_to_zero"]))
    best_mix = min(summaries, key=lambda row: abs(row["closest_mixing_w2"] - 0.5))
    return {
        "windows": summaries,
        "best_gap_window": best_gap,
        "best_mixing_window": best_mix,
    }


def clone_config(base: EVBConfig, delta_alpha: float, output_root: Path) -> EVBConfig:
    config = replace(base)
    output_dir = str(output_root / f"delta_alpha_{delta_alpha:+.0f}".replace("+", "p").replace("-", "m"))
    config.project = replace(
        base.project,
        name=f"{base.project.name}-da-{delta_alpha:+.0f}",
        output_dir=output_dir,
    )
    config.output_dir = output_dir
    config.evb_parameters = replace(base.evb_parameters, delta_alpha=delta_alpha)
    return config


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "delta_alpha_scan_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (root / "delta_alpha_scan_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "delta_alpha_kj_mol",
                "best_gap_window",
                "closest_gap_to_zero_kj_mol",
                "w2_at_closest_gap",
                "pt_at_closest_gap_nm",
                "ro_at_closest_gap_nm",
                "best_mixing_window",
                "closest_mixing_w2",
                "gap_at_closest_mixing_kj_mol",
                "pt_at_closest_mixing_nm",
                "ro_at_closest_mixing_nm",
            ]
        )
        for row in rows:
            if row.get("status") != "ok":
                writer.writerow([row["delta_alpha_kj_mol"], "FAILED", "", "", "", "", "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    row["delta_alpha_kj_mol"],
                    row["best_gap_window"]["window_id"],
                    row["best_gap_window"]["closest_gap_to_zero"],
                    row["best_gap_window"]["w2_at_closest_gap"],
                    row["best_gap_window"]["pt_at_closest_gap"],
                    row["best_gap_window"]["ro_at_closest_gap"],
                    row["best_mixing_window"]["window_id"],
                    row["best_mixing_window"]["closest_mixing_w2"],
                    row["best_mixing_window"]["gap_at_closest_mixing"],
                    row["best_mixing_window"]["pt_at_closest_mixing"],
                    row["best_mixing_window"]["ro_at_closest_mixing"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--delta-alpha-values", required=True, help="Comma-separated delta_alpha values in kJ/mol")
    parser.add_argument(
        "--output-root",
        default="outputs/delta_alpha_gap_native_scan",
        help="Directory where per-delta runs and summary files will be written.",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    output_root = Path(args.output_root)
    delta_values = [float(token.strip()) for token in args.delta_alpha_values.split(",") if token.strip()]
    rows: list[dict[str, Any]] = []

    for delta_alpha in delta_values:
        config = clone_config(base, delta_alpha, output_root)
        try:
            run_sample_series(config)
            summary = summarize_output(Path(config.output_dir))
            rows.append(
                {
                    "delta_alpha_kj_mol": delta_alpha,
                    "output_dir": config.output_dir,
                    "status": "ok",
                    "best_gap_window": summary["best_gap_window"],
                    "best_mixing_window": summary["best_mixing_window"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "delta_alpha_kj_mol": delta_alpha,
                    "output_dir": config.output_dir,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    write_summary(output_root, rows)


if __name__ == "__main__":
    main()
