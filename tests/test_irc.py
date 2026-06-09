from __future__ import annotations

from pathlib import Path

import pytest

from kemp_evb.irc import canonicalize_irc_path, identify_fixed_atoms, parse_reference_profile, read_irc_xyz, summarize_irc, write_irc_outputs


def test_read_irc_xyz_extracts_comment_energies_and_missing_ts(tmp_path: Path):
    xyz = tmp_path / "irc.xyz"
    xyz.write_text(
        """
2
 energy: -10.000000 gnorm: 0.1
H 0.0 0.0 0.0
H 0.0 0.0 0.7

2
TS/base mode=1 freq=-100.0
H 0.0 0.0 0.0
H 0.0 0.0 0.8

2
 energy: -9.990000 gnorm: 0.1
H 0.0 0.0 0.0
H 0.0 0.0 0.9
""",
        encoding="utf-8",
    )

    frames = read_irc_xyz(xyz)
    summary = summarize_irc(frames)

    assert len(frames) == 3
    assert frames[0].energy_hartree == -10.0
    assert frames[1].energy_hartree is None
    assert summary.missing_energy_frames == [1]
    assert summary.ts_comment_frames == [1]
    assert summary.maximum_energy_frame == 2


def test_read_irc_xyz_allows_comments_without_energies(tmp_path: Path):
    xyz = tmp_path / "irc_no_energy.xyz"
    xyz.write_text(
        """2
plain product-side geometry
H 0.0 0.0 0.0
H 0.0 0.0 0.7
2
plain reactant-side geometry
H 0.0 0.0 0.0
H 0.0 0.0 0.8
""",
        encoding="utf-8",
    )

    frames = read_irc_xyz(xyz)

    assert len(frames) == 2
    assert frames[0].energy_hartree is None
    assert frames[1].energy_hartree is None


def test_canonicalize_prod_ts_rc_reverses_to_reactant_first(tmp_path: Path):
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

    assert [frame.role for frame in path.frames] == ["RC", "TS", "PROD"]
    assert path.rc_frame == 0
    assert path.ts_frame == 1
    assert path.product_frame == 2
    assert path.canonical_to_original == {0: 2, 1: 1, 2: 0}


def test_reference_profile_converts_kcal_to_kj_and_relative_targets():
    profile = parse_reference_profile("kcal/mol", rc=1.0, ts=19.5, product=8.2)

    assert profile.rc_kj_mol == pytest.approx(4.184)
    assert profile.target_barrier_kj_mol == pytest.approx(18.5 * 4.184)
    assert profile.target_reaction_free_energy_kj_mol == pytest.approx(7.2 * 4.184)


def test_explicit_original_frames_override_auto_after_reversal(tmp_path: Path):
    xyz = tmp_path / "explicit.xyz"
    xyz.write_text(
        """1
unlabeled product
C 2.0 0.0 0.0
1
unlabeled ts
C 1.0 0.0 0.0
1
unlabeled reactant
C 0.0 0.0 0.0
""",
        encoding="utf-8",
    )

    path = canonicalize_irc_path(
        read_irc_xyz(xyz),
        order="prod_ts_rc",
        rc_frame=2,
        ts_frame=1,
        product_frame=0,
    )

    assert path.rc_frame == 0
    assert path.ts_frame == 1
    assert path.product_frame == 2
    assert path.original_to_canonical == {2: 0, 1: 1, 0: 2}


def test_write_irc_outputs_creates_profile_and_plots(tmp_path: Path):
    xyz = tmp_path / "irc.xyz"
    xyz.write_text(
        """1
 energy: -1.000
He 0.0 0.0 0.0
1
 energy: -0.990
He 0.1 0.0 0.0
""",
        encoding="utf-8",
    )
    frames = read_irc_xyz(xyz)
    output = tmp_path / "out"

    payload = write_irc_outputs(frames, output, title="test")

    assert payload["n_frames"] == 2
    assert (output / "irc_profile.csv").is_file()
    assert (output / "irc_summary.json").is_file()
    assert (output / "irc_energy_profile_kcal.png").is_file()
    assert (output / "fixed_carbon_candidates.yaml").is_file()


def test_identify_fixed_atoms_reports_only_stationary_atoms(tmp_path: Path):
    xyz = tmp_path / "irc.xyz"
    xyz.write_text(
        """2
 energy: -1.000
C 0.0 0.0 0.0
C 1.0 0.0 0.0
2
 energy: -0.990
C 0.0 0.0 0.0
C 1.2 0.0 0.0
""",
        encoding="utf-8",
    )
    frames = read_irc_xyz(xyz)

    fixed = identify_fixed_atoms(frames, element="C")

    assert [candidate.frame_atom_index for candidate in fixed] == [0]
