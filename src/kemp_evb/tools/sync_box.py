from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..openmm_backend import AmberSystemLoader


@dataclass(slots=True)
class BoxSyncResult:
    state1_prmtop: str
    state1_pdb: str
    state2_prmtop: str
    state2_pdb: str
    source: str
    box_lengths_angstrom: tuple[float, float, float]
    wrapped_coordinates: bool


def sync_shared_box(
    state1_prmtop: str | Path,
    state1_pdb: str | Path,
    state2_prmtop: str | Path,
    state2_pdb: str | Path,
    output_dir: str | Path,
    source: str = "state1",
    wrap_coordinates: bool = False,
) -> BoxSyncResult:
    state1_prmtop = Path(state1_prmtop)
    state1_pdb = Path(state1_pdb)
    state2_prmtop = Path(state2_prmtop)
    state2_pdb = Path(state2_pdb)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AmberSystemLoader(nonbonded_method="PME", constraints="None")
    loaded1 = loader.load(str(state1_prmtop), str(state1_pdb))
    loaded2 = loader.load(str(state2_prmtop), str(state2_pdb))
    if loaded1.box_vectors_nm is None or loaded2.box_vectors_nm is None:
        raise ValueError("Both states must define periodic box vectors to synchronize them.")

    if source == "state1":
        box_vectors_nm = loaded1.box_vectors_nm
    elif source == "state2":
        box_vectors_nm = loaded2.box_vectors_nm
    elif source == "average":
        box_vectors_nm = 0.5 * (loaded1.box_vectors_nm + loaded2.box_vectors_nm)
    else:
        raise ValueError("source must be one of: 'state1', 'state2', 'average'.")

    _validate_orthorhombic(box_vectors_nm)
    box_lengths_angstrom = tuple(float(box_vectors_nm[i, i] * 10.0) for i in range(3))

    state1_pdb_lines = _rewrite_pdb_cryst1(state1_pdb, box_lengths_angstrom, wrap_coordinates)
    state2_pdb_lines = _rewrite_pdb_cryst1(state2_pdb, box_lengths_angstrom, wrap_coordinates)
    state1_prmtop_lines = _rewrite_prmtop_box_dimensions(state1_prmtop, box_lengths_angstrom)
    state2_prmtop_lines = _rewrite_prmtop_box_dimensions(state2_prmtop, box_lengths_angstrom)

    out1_prmtop = output_dir / state1_prmtop.name
    out2_prmtop = output_dir / state2_prmtop.name
    out1_pdb = output_dir / state1_pdb.name
    out2_pdb = output_dir / state2_pdb.name
    out1_prmtop.write_text("".join(state1_prmtop_lines), encoding="utf-8")
    out2_prmtop.write_text("".join(state2_prmtop_lines), encoding="utf-8")
    out1_pdb.write_text("".join(state1_pdb_lines), encoding="utf-8")
    out2_pdb.write_text("".join(state2_pdb_lines), encoding="utf-8")

    return BoxSyncResult(
        state1_prmtop=str(out1_prmtop),
        state1_pdb=str(out1_pdb),
        state2_prmtop=str(out2_prmtop),
        state2_pdb=str(out2_pdb),
        source=source,
        box_lengths_angstrom=box_lengths_angstrom,
        wrapped_coordinates=wrap_coordinates,
    )


def _validate_orthorhombic(box_vectors_nm: np.ndarray) -> None:
    diagonal = np.diag(np.diag(box_vectors_nm))
    if not np.allclose(box_vectors_nm, diagonal, atol=1.0e-8):
        raise ValueError("Only orthorhombic periodic boxes are supported by this utility.")


def _rewrite_pdb_cryst1(path: Path, box_lengths_angstrom: tuple[float, float, float], wrap_coordinates: bool) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    a, b, c = box_lengths_angstrom
    cryst1 = f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1\n"
    rewritten: list[str] = []
    saw_cryst1 = False
    for line in lines:
        record = line[:6].strip()
        if record == "CRYST1":
            rewritten.append(cryst1)
            saw_cryst1 = True
            continue
        if wrap_coordinates and record in {"ATOM", "HETATM"}:
            x = _wrap_coordinate(float(line[30:38]), a)
            y = _wrap_coordinate(float(line[38:46]), b)
            z = _wrap_coordinate(float(line[46:54]), c)
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        rewritten.append(line)
    if not saw_cryst1:
        insertion_index = 0
        if rewritten and rewritten[0].startswith("HEADER"):
            insertion_index = 1
        rewritten.insert(insertion_index, cryst1)
    return rewritten


def _wrap_coordinate(value: float, box_length: float) -> float:
    wrapped = value % box_length
    if wrapped >= box_length:
        wrapped -= box_length
    return wrapped


def _rewrite_prmtop_box_dimensions(path: Path, box_lengths_angstrom: tuple[float, float, float]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    flag_idx = next((i for i, line in enumerate(lines) if line.startswith("%FLAG BOX_DIMENSIONS")), None)
    if flag_idx is None:
        raise ValueError(f"%FLAG BOX_DIMENSIONS block not found in {path}")
    data_start = flag_idx + 2
    data_end = next((i for i in range(data_start, len(lines)) if lines[i].startswith("%FLAG ")), len(lines))
    alpha = 90.0
    a, b, c = box_lengths_angstrom
    payload = [alpha, a, b, c]
    rebuilt = "".join(f"{value:16.8E}" for value in payload) + "\n"
    return lines[:data_start] + [rebuilt] + lines[data_end:]
