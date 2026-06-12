from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .openmm_backend import load_positions_file
from .reference_profile import canonical_label
from .irc import read_irc_xyz


@dataclass(slots=True)
class FrameMappingResult:
    frames: dict[str, np.ndarray]
    report: list[dict[str, Any]]


def collect_reaction_frames(config_payload: dict[str, Any], *, expected_atom_count: int | None = None) -> FrameMappingResult:
    frames: dict[str, np.ndarray] = {}
    report: list[dict[str, Any]] = []
    for label, row in (config_payload.get("frames") or {}).items():
        canon = canonical_label(label)
        path = Path(row.get("coordinates") if isinstance(row, dict) else row)
        positions = load_positions_file(str(path))
        _check_count(path, positions, expected_atom_count)
        frames[canon] = positions
        report.append({"label": canon, "source_path": str(path), "coordinate_format": path.suffix.lstrip("."), "full_system_coordinates_available": True, "can_be_used_for_calibration": True})
    irc = config_payload.get("irc") or {}
    if irc.get("path") and irc.get("frame_labels"):
        irc_frames = read_irc_xyz(irc["path"])
        for label, idx in irc["frame_labels"].items():
            canon = canonical_label(label)
            frame = irc_frames[int(idx)]
            coords = frame.coordinates_nm
            if expected_atom_count is not None and len(coords) != expected_atom_count:
                raise ValueError(f"IRC frame {label!r} has {len(coords)} atoms, expected full-system count {expected_atom_count}; provide embedded full-system coordinates.")
            frames[canon] = coords
            report.append({"label": canon, "source_path": str(irc["path"]), "coordinate_format": "xyz", "frame_index": int(idx), "full_system_coordinates_available": expected_atom_count is None or len(coords) == expected_atom_count, "can_be_used_for_calibration": True})
    return FrameMappingResult(frames, report)


def _check_count(path: Path, positions: np.ndarray, expected_atom_count: int | None) -> None:
    if expected_atom_count is not None and len(positions) != expected_atom_count:
        raise ValueError(f"Coordinate file {path} has {len(positions)} atoms, expected {expected_atom_count}.")
