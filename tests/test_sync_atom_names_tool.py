from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("parmed")

from kemp_evb.tools import sync_atom_names
from kemp_evb.openmm_backend import AmberSystemLoader


def test_sync_atom_names_tool_on_real_like_inputs(tmp_path: Path):
    outdir = tmp_path / "synced"
    result = sync_atom_names(
        state1_prmtop="systems/KEMP-solvent/RC.prmtop",
        state1_pdb="systems/KEMP-solvent/RC.pdb",
        state2_prmtop="systems/KEMP-solvent/PROD.prmtop",
        state2_pdb="systems/KEMP-solvent/PROD.pdb",
        output_dir=outdir,
    )
    assert result.renamed_indices
    assert all(name.startswith("E") for name in result.assigned_names)
    loader = AmberSystemLoader(nonbonded_method="PME", constraints="HBonds")
    state1 = loader.load(result.state1_prmtop, result.state1_pdb)
    state2 = loader.load(result.state2_prmtop, result.state2_pdb)
    assert state1.atom_names == state2.atom_names
