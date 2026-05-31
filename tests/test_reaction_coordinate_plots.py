from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from kemp_evb.analysis.reaction_coordinate_plots import load_reaction_coordinate_frames, write_reaction_coordinate_plots


def _write_window_csv(path: Path, window_id: str, branch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "step",
                "time_ps",
                "window_id",
                "rc_center_nm",
                "E1_kj_mol",
                "E2_kj_mol",
                "delta_e_kj_mol",
                "delta_e_shifted_kj_mol",
                "Eevb_kj_mol",
                "w1",
                "w2",
                "proton_transfer_rc_nm",
                "proton_transfer_event",
                "proton_transfer_rc",
                "ring_opening_rc",
            ]
        )
        writer.writerow([0, 100, 0.1, window_id, 0.0, -1.0, -2.0, 1.0, 1.0, -2.5, 0.5, 0.5, -0.05, 0, -0.05, 0.14])
        writer.writerow([1, 200, 0.2, window_id, 0.0, -1.0, -2.0, 1.2, 1.2, -2.4, 0.5, 0.5, -0.02, 0, -0.02, 0.16])


def test_loads_reaction_coordinate_frames_and_writes_plot(tmp_path: Path):
    pytest.importorskip("matplotlib")
    _write_window_csv(tmp_path / "branches" / "forward" / "windows" / "p024" / "production_observables.csv", "p024", "forward")
    _write_window_csv(tmp_path / "branches" / "reverse" / "windows" / "p024" / "production_observables.csv", "p024", "reverse")
    qm_guide = {
        "structures": {
            "RC": {"proton_transfer_rc_angstrom": -1.0, "ring_opening_rc_angstrom": 1.4},
            "TS": {"proton_transfer_rc_angstrom": 0.5, "ring_opening_rc_angstrom": 1.58},
            "PROD": {"proton_transfer_rc_angstrom": 2.0, "ring_opening_rc_angstrom": 3.5},
        }
    }
    guide_path = tmp_path / "qm_geometry_guide.json"
    guide_path.write_text(json.dumps(qm_guide), encoding="utf-8")

    frames = load_reaction_coordinate_frames(tmp_path)
    assert len(frames) == 4
    csv_path, png_path, summary_path = write_reaction_coordinate_plots(tmp_path, qm_geometry_guide=guide_path)
    assert csv_path.exists()
    assert png_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "forward" in summary["branches"]
    assert "reverse" in summary["branches"]
    assert "qm_points_nm" in summary
