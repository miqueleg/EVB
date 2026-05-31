from __future__ import annotations

import argparse
import math
from pathlib import Path


def _read_pdb(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _atom_records(lines: list[str]) -> list[tuple[int, str, str, str, int, tuple[float, float, float], str]]:
    records = []
    atom_index = 0
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_index += 1
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip()
        resid = int(line[22:26])
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        records.append((atom_index, atom_name, resname, chain, resid, xyz, line))
    return records


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-pdb", required=True)
    parser.add_argument("--rc-reactive-pdb", required=True)
    parser.add_argument("--prod-reactive-pdb", required=True)
    parser.add_argument("--reactive-atom-count", type=int, required=True)
    parser.add_argument("--cutoff-angstrom", type=float, default=1.8)
    parser.add_argument("--output-pdb", required=True)
    args = parser.parse_args()

    template_lines = _read_pdb(Path(args.template_pdb))
    rc_lines = _read_pdb(Path(args.rc_reactive_pdb))
    prod_lines = _read_pdb(Path(args.prod_reactive_pdb))

    template_atoms = _atom_records(template_lines)
    rc_atoms = _atom_records(rc_lines)[: args.reactive_atom_count]
    prod_atoms = _atom_records(prod_lines)[: args.reactive_atom_count]
    reactive_xyz = [rec[5] for rec in rc_atoms] + [rec[5] for rec in prod_atoms]

    strip_residues: set[tuple[str, int, str]] = set()
    for _, _, resname, chain, resid, xyz, _ in template_atoms:
        if resname not in {"WAT", "HOH"}:
            continue
        if any(_dist(xyz, ref) < args.cutoff_angstrom for ref in reactive_xyz):
            strip_residues.add((resname, resid, chain))

    out_lines: list[str] = []
    for line in template_lines:
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            chain = line[21].strip()
            resid = int(line[22:26])
            if (resname, resid, chain) in strip_residues:
                continue
        out_lines.append(line)

    Path(args.output_pdb).write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
