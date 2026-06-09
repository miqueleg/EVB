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
    inputs = [
        Path("outputs/prepared_systems/kemp_solvent_synced/RC.prmtop"),
        Path("outputs/prepared_systems/kemp_solvent_synced/RC.pdb"),
        Path("outputs/prepared_systems/kemp_solvent_synced/PROD.prmtop"),
        Path("outputs/prepared_systems/kemp_solvent_synced/PROD.pdb"),
    ]
    if not all(path.exists() for path in inputs):
        pytest.skip("optional generated AMBER sync-box fixture is not present")

    outdir = tmp_path / "boxed"
    result = sync_shared_box(
        state1_prmtop=str(inputs[0]),
        state1_pdb=str(inputs[1]),
        state2_prmtop=str(inputs[2]),
        state2_pdb=str(inputs[3]),
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
