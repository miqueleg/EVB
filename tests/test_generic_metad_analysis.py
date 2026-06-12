from __future__ import annotations

from pathlib import Path

import pytest

from kemp_evb.metad_analysis import reconstruct_wt_fel_from_bias_table


def test_wt_metad_bias_table_reconstruction(tmp_path: Path):
    table = tmp_path / "bias_table.csv"
    table.write_text("gap_kj_mol,bias_kj_mol\n-1,0\n0,-2\n1,-4\n", encoding="utf-8")

    result = reconstruct_wt_fel_from_bias_table(table, tmp_path / "analysis", bias_factor=5.0)

    assert result["n_grid_points"] == 3
    assert (tmp_path / "analysis" / "metad_fel.csv").exists()


def test_wt_metad_requires_valid_bias_factor(tmp_path: Path):
    table = tmp_path / "bias_table.csv"
    table.write_text("gap_kj_mol,bias_kj_mol\n0,0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        reconstruct_wt_fel_from_bias_table(table, tmp_path / "analysis", bias_factor=1.0)
