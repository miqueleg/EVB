from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..openmm_backend import AmberSystemLoader


@dataclass(slots=True)
class SyncResult:
    state1_prmtop: str
    state1_pdb: str
    state2_prmtop: str
    state2_pdb: str
    renamed_indices: list[int]
    assigned_names: list[str]


def sync_atom_names(
    state1_prmtop: str | Path,
    state1_pdb: str | Path,
    state2_prmtop: str | Path,
    state2_pdb: str | Path,
    output_dir: str | Path,
    prefix: str = "E",
) -> SyncResult:
    state1_prmtop = Path(state1_prmtop)
    state1_pdb = Path(state1_pdb)
    state2_prmtop = Path(state2_prmtop)
    state2_pdb = Path(state2_pdb)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AmberSystemLoader(nonbonded_method="PME", constraints="HBonds")
    loaded1 = loader.load(str(state1_prmtop), str(state1_pdb))
    loaded2 = loader.load(str(state2_prmtop), str(state2_pdb))
    if len(loaded1.atom_names) != len(loaded2.atom_names):
        raise ValueError("State 1 and state 2 have different atom counts.")

    names1 = list(loaded1.atom_names)
    names2 = list(loaded2.atom_names)
    masses1 = list(map(float, loaded1.masses_amu))
    masses2 = list(map(float, loaded2.masses_amu))
    mismatches = [index for index, (name1, name2) in enumerate(zip(names1, names2)) if name1 != name2]
    bad_mass = [index for index, (mass1, mass2) in enumerate(zip(masses1, masses2)) if abs(mass1 - mass2) > 1.0e-6]
    if bad_mass:
        raise ValueError(f"Cannot synchronize names because masses still differ at indices: {bad_mass[:10]}")

    used_names = {name.strip() for name in names1} | {name.strip() for name in names2}
    replacements = {}
    for counter, index in enumerate(mismatches, start=1):
        new_name = _next_available_name(used_names, prefix=prefix, start=counter)
        used_names.add(new_name)
        replacements[index] = new_name

    state1_pdb_lines = _rewrite_pdb_atom_names(state1_pdb, replacements)
    state2_pdb_lines = _rewrite_pdb_atom_names(state2_pdb, replacements)
    state1_prmtop_lines = _rewrite_prmtop_atom_names(state1_prmtop, replacements)
    state2_prmtop_lines = _rewrite_prmtop_atom_names(state2_prmtop, replacements)

    out1_prmtop = output_dir / state1_prmtop.name
    out2_prmtop = output_dir / state2_prmtop.name
    out1_pdb = output_dir / state1_pdb.name
    out2_pdb = output_dir / state2_pdb.name
    out1_prmtop.write_text("".join(state1_prmtop_lines), encoding="utf-8")
    out2_prmtop.write_text("".join(state2_prmtop_lines), encoding="utf-8")
    out1_pdb.write_text("".join(state1_pdb_lines), encoding="utf-8")
    out2_pdb.write_text("".join(state2_pdb_lines), encoding="utf-8")

    return SyncResult(
        state1_prmtop=str(out1_prmtop),
        state1_pdb=str(out1_pdb),
        state2_prmtop=str(out2_prmtop),
        state2_pdb=str(out2_pdb),
        renamed_indices=sorted(replacements),
        assigned_names=[replacements[index] for index in sorted(replacements)],
    )


def _next_available_name(used_names: set[str], prefix: str, start: int) -> str:
    counter = start
    while True:
        candidate = f"{prefix}{counter:03d}"[-4:]
        if candidate not in used_names:
            return candidate
        counter += 1


def _rewrite_pdb_atom_names(path: Path, replacements: dict[int, str]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    atom_line_index = 0
    rewritten: list[str] = []
    for line in lines:
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            if atom_line_index in replacements:
                name = f"{replacements[atom_line_index]:>4s}"
                line = f"{line[:12]}{name}{line[16:]}"
            atom_line_index += 1
        rewritten.append(line)
    if atom_line_index == 0:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return rewritten


def _rewrite_prmtop_atom_names(path: Path, replacements: dict[int, str]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    flag_idx = next((i for i, line in enumerate(lines) if line.startswith("%FLAG ATOM_NAME")), None)
    if flag_idx is None:
        raise ValueError(f"%FLAG ATOM_NAME block not found in {path}")
    format_idx = flag_idx + 1
    data_start = flag_idx + 2
    data_end = next((i for i in range(data_start, len(lines)) if lines[i].startswith("%FLAG ")), len(lines))
    atom_name_block = "".join(line.rstrip("\n") for line in lines[data_start:data_end])
    names = [atom_name_block[i : i + 4] for i in range(0, len(atom_name_block), 4)]
    for index, new_name in replacements.items():
        names[index] = f"{new_name:<4s}"[:4]
    rebuilt = ["".join(names[i : i + 20]) + "\n" for i in range(0, len(names), 20)]
    return lines[:data_start] + rebuilt + lines[data_end:]
