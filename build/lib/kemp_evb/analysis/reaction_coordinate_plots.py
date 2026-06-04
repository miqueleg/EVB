from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ReactionCoordinateFrame:
    branch: str
    window_id: str
    frame: int
    proton_transfer_rc_nm: float
    ring_opening_rc_nm: float
    delta_e_shifted_kj_mol: float
    evb_energy_kj_mol: float


def load_reaction_coordinate_frames(output_dir: str | Path) -> list[ReactionCoordinateFrame]:
    root = Path(output_dir)
    csv_paths: list[tuple[str, Path]] = []
    direct_root = root / "windows"
    if direct_root.exists():
        csv_paths.extend(("main", path) for path in sorted(direct_root.glob("*/production_observables.csv")))
    branches_root = root / "branches"
    if branches_root.exists():
        for branch_dir in sorted(branches_root.iterdir()):
            if not branch_dir.is_dir():
                continue
            csv_paths.extend((branch_dir.name, path) for path in sorted((branch_dir / "windows").glob("*/production_observables.csv")))
    frames: list[ReactionCoordinateFrame] = []
    for branch, csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("proton_transfer_rc") or not row.get("ring_opening_rc"):
                    continue
                frames.append(
                    ReactionCoordinateFrame(
                        branch=branch,
                        window_id=row["window_id"],
                        frame=int(row["frame"]),
                        proton_transfer_rc_nm=float(row["proton_transfer_rc"]),
                        ring_opening_rc_nm=float(row["ring_opening_rc"]),
                        delta_e_shifted_kj_mol=float(row["delta_e_shifted_kj_mol"]),
                        evb_energy_kj_mol=float(row["Eevb_kj_mol"]),
                    )
                )
    if not frames:
        raise ValueError(f"No reaction-coordinate frames with proton_transfer_rc and ring_opening_rc were found under {root}")
    return frames


def write_reaction_coordinate_plots(
    output_dir: str | Path,
    *,
    qm_geometry_guide: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    import matplotlib.pyplot as plt

    frames = load_reaction_coordinate_frames(output_dir)
    destination = Path(output_dir) / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "reaction_coordinate_2d.csv"
    png_path = destination / "reaction_coordinate_2d.png"
    summary_path = destination / "reaction_coordinate_2d_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "window_id",
                "frame",
                "proton_transfer_rc_nm",
                "ring_opening_rc_nm",
                "delta_e_shifted_kj_mol",
                "evb_energy_kj_mol",
            ]
        )
        for frame in frames:
            writer.writerow(
                [
                    frame.branch,
                    frame.window_id,
                    frame.frame,
                    frame.proton_transfer_rc_nm,
                    frame.ring_opening_rc_nm,
                    frame.delta_e_shifted_kj_mol,
                    frame.evb_energy_kj_mol,
                ]
            )

    plt.figure(figsize=(6.8, 5.2), dpi=220)
    colors = {"main": "0.25", "forward": "#1f77b4", "reverse": "#d95f02"}
    grouped: dict[str, list[ReactionCoordinateFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.branch, []).append(frame)
    summary: dict[str, object] = {"branches": {}, "qm_geometry_guide": str(qm_geometry_guide) if qm_geometry_guide else None}
    for branch, branch_frames in sorted(grouped.items()):
        x = np.asarray([item.proton_transfer_rc_nm for item in branch_frames], dtype=float)
        y = np.asarray([item.ring_opening_rc_nm for item in branch_frames], dtype=float)
        plt.scatter(x, y, s=10, alpha=0.35, color=colors.get(branch, "0.5"), label=f"{branch} frames")
        window_ids = sorted({item.window_id for item in branch_frames})
        mean_x: list[float] = []
        mean_y: list[float] = []
        for window_id in window_ids:
            members = [item for item in branch_frames if item.window_id == window_id]
            mean_x.append(float(np.mean([item.proton_transfer_rc_nm for item in members])))
            mean_y.append(float(np.mean([item.ring_opening_rc_nm for item in members])))
        plt.plot(mean_x, mean_y, color=colors.get(branch, "0.5"), linewidth=1.5, alpha=0.9)
        summary["branches"][branch] = {
            "n_frames": len(branch_frames),
            "n_windows": len(window_ids),
            "proton_transfer_rc_nm_range": [float(np.min(x)), float(np.max(x))],
            "ring_opening_rc_nm_range": [float(np.min(y)), float(np.max(y))],
        }

    if qm_geometry_guide is not None:
        guide_path = Path(qm_geometry_guide)
        if guide_path.exists():
            guide = json.loads(guide_path.read_text(encoding="utf-8"))
            structures = guide.get("structures", {})
            qm_points: dict[str, dict[str, float]] = {}
            markers = {"RC": "o", "TS": "X", "PROD": "s"}
            for label in ("RC", "TS", "PROD"):
                data = structures.get(label)
                if not data:
                    continue
                x = float(data["proton_transfer_rc_angstrom"]) * 0.1
                y = float(data["ring_opening_rc_angstrom"]) * 0.1
                qm_points[label] = {"proton_transfer_rc_nm": x, "ring_opening_rc_nm": y}
                plt.scatter([x], [y], s=80, marker=markers.get(label, "o"), color="black")
                plt.text(x + 0.005, y + 0.005, label, fontsize=8)
            summary["qm_points_nm"] = qm_points

    plt.xlabel("Proton Transfer RC / nm")
    plt.ylabel("N-O Ring Opening RC / nm")
    plt.title("EVB Reaction-Coordinate Landscape")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, png_path, summary_path
