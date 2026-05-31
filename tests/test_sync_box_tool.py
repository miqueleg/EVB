from __future__ import annotations

from pathlib import Path

import pytest

from kemp_evb.openmm_backend import AmberSystemLoader
from kemp_evb.tools import sync_shared_box


def _read_cryst1_lengths(path: Path) -> tuple[float, float, float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                return float(line[6:15]), float(line[15:24]), float(line[24:33])
    raise AssertionError(f"No CRYST1 record found in {path}")


def test_sync_shared_box_on_real_like_inputs(tmp_path: Path):
    outdir = tmp_path / "boxed"
    result = sync_shared_box(
        state1_prmtop="outputs/prepared_systems/kemp_solvent_synced/RC.prmtop",
        state1_pdb="outputs/prepared_systems/kemp_solvent_synced/RC.pdb",
        state2_prmtop="outputs/prepared_systems/kemp_solvent_synced/PROD.prmtop",
        state2_pdb="outputs/prepared_systems/kemp_solvent_synced/PROD.pdb",
        output_dir=outdir,
        source="state1",
    )
    assert result.source == "state1"
    cryst1_state1 = _read_cryst1_lengths(Path(result.state1_pdb))
    cryst1_state2 = _read_cryst1_lengths(Path(result.state2_pdb))
    assert cryst1_state1 == pytest.approx(cryst1_state2)

    loader = AmberSystemLoader(nonbonded_method="PME", constraints="None")
    state1 = loader.load(result.state1_prmtop, result.state1_pdb)
    state2 = loader.load(result.state2_prmtop, result.state2_pdb)
    assert state1.box_vectors_nm is not None
    assert state2.box_vectors_nm is not None
    assert state1.box_vectors_nm == pytest.approx(state2.box_vectors_nm)
