from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kemp_evb.config import EVBConfig, StateFiles
from kemp_evb.irc import canonicalize_irc_path, parse_reference_profile, read_irc_xyz
from kemp_evb.irc_setup import (
    _apply_explicit_mapping,
    _coerce_explicit_irc_mapping,
    _mixing_focused_gap_centers,
    _selected_relaxation_frames,
    _ts_focused_gap_centers,
    _write_gap_window_proposal,
    fit_evb_from_irc_roles,
)


def test_irc_fit_role_mapping_keeps_rc_and_product_distinct_when_reversed(tmp_path: Path, monkeypatch):
    xyz = tmp_path / "reversed.xyz"
    xyz.write_text(
        """1
PROD
H 2.0 0.0 0.0
1
TS
H 1.0 0.0 0.0
1
RC
H 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    path = canonicalize_irc_path(read_irc_xyz(xyz), order="prod_ts_rc")
    rows = [
        {"canonical_frame": 0, "E1_kj_mol": 10.0, "E2_kj_mol": 110.0},
        {"canonical_frame": 1, "E1_kj_mol": 60.0, "E2_kj_mol": 70.0},
        {"canonical_frame": 2, "E1_kj_mol": 120.0, "E2_kj_mol": 20.0},
    ]
    captured = {}

    def fake_fit(**kwargs):
        captured.update(kwargs)
        return "fit-result"

    monkeypatch.setattr("kemp_evb.irc_setup.fit_evb_reference_profile", fake_fit)

    result = fit_evb_from_irc_roles(path, rows, parse_reference_profile("kcal/mol", 0.0, 18.5, 7.2))

    assert result == "fit-result"
    assert captured["e_mm_min1_state1"] == 10.0
    assert captured["e_mm_min1_state2"] == 110.0
    assert captured["e_mm_min2_state1"] == 120.0
    assert captured["e_mm_min2_state2"] == 20.0
    assert captured["e_mm_ts_state1"] == 60.0
    assert captured["e_mm_ts_state2"] == 70.0
    assert captured["e_qmmm_min1"] == 0.0
    assert captured["e_qmmm_min2"] == 7.2 * 4.184
    assert captured["e_qmmm_ts"] == 18.5 * 4.184


def test_explicit_irc_to_openmm_mapping_overrides_auto_assignments():
    atoms = [
        SimpleNamespace(index=0, name="C1", element=SimpleNamespace(symbol="C")),
        SimpleNamespace(index=1, name="N1", element=SimpleNamespace(symbol="N")),
        SimpleNamespace(index=2, name="H1", element=SimpleNamespace(symbol="H")),
    ]
    explicit = _coerce_explicit_irc_mapping({"3": 0, "4": 1})

    mapping = _apply_explicit_mapping({0: 0, 1: 1, 2: 2}, explicit, ["C", "N", "H", "C", "N"], atoms)

    assert mapping[3] == 0
    assert mapping[4] == 1
    assert 0 not in mapping
    assert 1 not in mapping


def test_pathological_irc_scan_blocks_window_generation(tmp_path: Path):
    xyz = tmp_path / "path.xyz"
    xyz.write_text(
        """1
RC
H 0 0 0
1
TS
H 0 0 1
1
PROD
H 0 0 2
""",
        encoding="utf-8",
    )
    path = canonicalize_irc_path(read_irc_xyz(xyz), order="rc_ts_prod")
    rows = [
        {"gap_shifted_kj_mol": -100.0},
        {"gap_shifted_kj_mol": 50000.0},
        {"gap_shifted_kj_mol": 90000.0},
    ]
    config = EVBConfig(
        state1=StateFiles(prmtop="a", inpcrd="b"),
        state2=StateFiles(prmtop="c", inpcrd="d"),
    )

    payload = _write_gap_window_proposal(config, path, rows, tmp_path)

    assert payload["status"] == "blocked_pathological_irc_scan"
    assert payload["windows"] == []


def test_ts_focused_gap_centers_put_transition_region_near_middle():
    centers = _ts_focused_gap_centers(endpoint1=-30000.0, ts_gap=-1000.0, endpoint2=3000.0, n_windows=40)

    assert len(centers) == 40
    assert centers[0] == -30000.0
    assert centers[-1] == 3000.0
    assert centers[20] == -1000.0
    assert abs(centers[20] - centers[19]) < abs(centers[1] - centers[0])
    assert abs(centers[21] - centers[20]) < abs(centers[-1] - centers[-2])


def test_mixing_focused_gap_centers_put_evb_mixing_near_middle_and_extend_basins():
    centers = _mixing_focused_gap_centers(endpoint1=-30000.0, endpoint2=3000.0, n_windows=64, mixing_gap=0.0, basin_extension=1000.0)

    assert len(centers) == 64
    assert centers[0] == -31000.0
    assert centers[32] == 0.0
    assert centers[-1] == 4000.0
    assert abs(abs(centers[32] - centers[31]) - abs(centers[1] - centers[0])) < 1.0e-12
    assert abs(abs(centers[33] - centers[32]) - abs(centers[-1] - centers[-2])) < 1.0e-12


def test_relaxation_frame_selection_includes_roles(tmp_path: Path):
    xyz = tmp_path / "path.xyz"
    xyz.write_text(
        """1
RC
H 0 0 0
1
mid
H 0 0 1
1
TS
H 0 0 2
1
mid
H 0 0 3
1
PROD
H 0 0 4
""",
        encoding="utf-8",
    )
    path = canonicalize_irc_path(read_irc_xyz(xyz), order="rc_ts_prod")

    assert _selected_relaxation_frames(path, stride=3, explicit=[]) == [0, 2, 3, 4]
    assert _selected_relaxation_frames(path, stride=1, explicit=[1, 4]) == [1, 4]
