from __future__ import annotations

from pathlib import Path

from kemp_evb.irc import identify_fixed_atoms, read_irc_xyz, summarize_irc, write_irc_outputs


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
