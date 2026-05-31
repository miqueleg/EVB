from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


ELECTRON_CHARGE_SCALE = 18.2223


@dataclass(slots=True)
class AtomRecord:
    index: int
    residue: str
    name: str
    atomic_number: int
    mass_amu: float
    amber_type: str
    type_index: int
    charge_e: float


@dataclass(slots=True)
class BondRecord:
    atom1: int
    atom2: int
    name1: str
    name2: str
    equilibrium_angstrom: float
    force_constant_kcal_mol_a2: float


def parse_prmtop(path: Path) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    flags: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("%FLAG "):
            name = lines[i][6:].strip()
            i += 2
            vals: list[str] = []
            while i < len(lines) and not lines[i].startswith("%FLAG "):
                if not lines[i].startswith("%FORMAT"):
                    vals.extend(lines[i].split())
                i += 1
            flags[name] = vals
        else:
            i += 1
    return flags


def residue_map(flags: dict[str, list[str]], n_atoms: int) -> list[str]:
    labels = flags["RESIDUE_LABEL"]
    ptr = [int(x) for x in flags["RESIDUE_POINTER"]]
    out: list[str] = []
    for idx, start in enumerate(ptr):
        end = ptr[idx + 1] - 1 if idx + 1 < len(ptr) else n_atoms
        out.extend([labels[idx]] * (end - start + 1))
    return out[:n_atoms]


def atom_records(flags: dict[str, list[str]], n_atoms: int) -> list[AtomRecord]:
    residues = residue_map(flags, len(flags["ATOM_NAME"]))
    records: list[AtomRecord] = []
    for i in range(n_atoms):
        records.append(
            AtomRecord(
                index=i,
                residue=residues[i],
                name=flags["ATOM_NAME"][i],
                atomic_number=int(flags["ATOMIC_NUMBER"][i]),
                mass_amu=float(flags["MASS"][i]),
                amber_type=flags["AMBER_ATOM_TYPE"][i],
                type_index=int(flags["ATOM_TYPE_INDEX"][i]),
                charge_e=float(flags["CHARGE"][i]) / ELECTRON_CHARGE_SCALE,
            )
        )
    return records


def bond_records(flags: dict[str, list[str]], atom_names: list[str], n_atoms: int) -> list[BondRecord]:
    eq = [float(x) for x in flags["BOND_EQUIL_VALUE"]]
    k = [float(x) for x in flags["BOND_FORCE_CONSTANT"]]
    out: list[BondRecord] = []
    for key in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        vals = [int(x) for x in flags[key]]
        for i in range(0, len(vals), 3):
            a = vals[i] // 3
            b = vals[i + 1] // 3
            t = vals[i + 2] - 1
            if a < n_atoms and b < n_atoms:
                out.append(
                    BondRecord(
                        atom1=a,
                        atom2=b,
                        name1=atom_names[a],
                        name2=atom_names[b],
                        equilibrium_angstrom=eq[t],
                        force_constant_kcal_mol_a2=k[t],
                    )
                )
    return sorted(out, key=lambda x: (x.atom1, x.atom2))


def summarize(path: Path, n_atoms: int = 23) -> dict:
    flags = parse_prmtop(path)
    atoms = atom_records(flags, n_atoms)
    bonds = bond_records(flags, [a.name for a in atoms], n_atoms)
    return {
        "path": str(path),
        "atoms": [asdict(a) for a in atoms],
        "bonds": [asdict(b) for b in bonds],
        "total_charge_e": sum(a.charge_e for a in atoms),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit reactive-region atom types, charges, and bonded terms in two Amber prmtops.")
    parser.add_argument("--state1", required=True)
    parser.add_argument("--state2", required=True)
    parser.add_argument("--n-atoms", type=int, default=23)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = {
        "state1": summarize(Path(args.state1), args.n_atoms),
        "state2": summarize(Path(args.state2), args.n_atoms),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
