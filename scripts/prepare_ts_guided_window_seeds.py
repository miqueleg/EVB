from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kemp_evb.config import load_config
from kemp_evb.observables import compute_named_reaction_coordinates
from kemp_evb.openmm_backend import load_positions_file


@dataclass(slots=True)
class SeedCandidate:
    branch: str
    window_id: str
    coordinates: str
    proton_transfer_rc_nm: float
    ring_opening_rc_nm: float
    ts_distance_nm: float


def _iter_final_states(output_dir: Path) -> list[tuple[str, str, Path]]:
    results: list[tuple[str, str, Path]] = []
    direct_root = output_dir / "windows"
    if direct_root.exists():
        for path in sorted(direct_root.glob("*/final_state.pdb")):
            results.append(("main", path.parent.name, path))
    branches_root = output_dir / "branches"
    if branches_root.exists():
        for branch_dir in sorted(branches_root.iterdir()):
            if not branch_dir.is_dir():
                continue
            for path in sorted((branch_dir / "windows").glob("*/final_state.pdb")):
                results.append((branch_dir.name, path.parent.name, path))
    return results


def _load_ts_target(qm_guide_path: Path) -> tuple[float, float]:
    guide = json.loads(qm_guide_path.read_text(encoding="utf-8"))
    ts = guide["structures"]["TS"]
    return float(ts["proton_transfer_rc_angstrom"]) * 0.1, float(ts["ring_opening_rc_angstrom"]) * 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qm-guide", default="systems/KEMP_implicit/qm_geometry_guide.json")
    parser.add_argument("--forward-windows", default="")
    parser.add_argument("--reverse-windows", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    ts_pt_nm, ts_ring_nm = _load_ts_target(Path(args.qm_guide))
    definitions = config.observables.reaction_coordinates
    if not definitions:
        raise ValueError("Config must define observables.reaction_coordinates before preparing TS-guided seeds.")

    candidates: list[SeedCandidate] = []
    for branch, window_id, path in _iter_final_states(Path(args.output_dir)):
        positions_nm = load_positions_file(str(path))
        values = compute_named_reaction_coordinates(positions_nm, definitions)
        if "proton_transfer_rc" not in values or "ring_opening_rc" not in values:
            continue
        dx = values["proton_transfer_rc"] - ts_pt_nm
        dy = values["ring_opening_rc"] - ts_ring_nm
        candidates.append(
            SeedCandidate(
                branch=branch,
                window_id=window_id,
                coordinates=str(path),
                proton_transfer_rc_nm=values["proton_transfer_rc"],
                ring_opening_rc_nm=values["ring_opening_rc"],
                ts_distance_nm=float(np.sqrt(dx * dx + dy * dy)),
            )
        )
    if not candidates:
        raise ValueError(f"No final_state.pdb candidates were found under {args.output_dir}")

    candidates.sort(key=lambda item: item.ts_distance_nm)
    by_branch: dict[str, list[SeedCandidate]] = {}
    for candidate in candidates:
        by_branch.setdefault(candidate.branch, []).append(candidate)

    destination = Path(args.output_dir) / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ts_seed_candidates.json").write_text(
        json.dumps(
            {
                "qm_guide": str(Path(args.qm_guide)),
                "ts_target_nm": {"proton_transfer_rc": ts_pt_nm, "ring_opening_rc": ts_ring_nm},
                "candidates": [asdict(item) for item in candidates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    forward_windows = [item.strip() for item in args.forward_windows.split(",") if item.strip()]
    reverse_windows = [item.strip() for item in args.reverse_windows.split(",") if item.strip()]
    lines = ["sampling:", "  seed_windows:"]
    if forward_windows:
        forward_choice = by_branch.get("forward", candidates)[0]
        for window_id in forward_windows:
            lines.extend(
                [
                    f"    - window_id: {window_id}",
                    f"      coordinates: {forward_choice.coordinates}",
                    "      branch: forward",
                ]
            )
    if reverse_windows:
        reverse_choice = by_branch.get("reverse", candidates)[0]
        for window_id in reverse_windows:
            lines.extend(
                [
                    f"    - window_id: {window_id}",
                    f"      coordinates: {reverse_choice.coordinates}",
                    "      branch: reverse",
                ]
            )
    if len(lines) == 2:
        lines.append("    []")
    (destination / "ts_seed_windows.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
