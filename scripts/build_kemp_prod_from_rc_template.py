from __future__ import annotations

import argparse
from pathlib import Path


def _read_pdb_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _extract_atom_records(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith(("ATOM", "HETATM"))]


def _replace_reactive_region(template_lines: list[str], reactive_atom_lines: list[str], reactive_atom_count: int) -> list[str]:
    output: list[str] = []
    atom_index = 0
    reactive_index = 0
    for line in template_lines:
        if line.startswith(("ATOM", "HETATM")):
            if atom_index < reactive_atom_count:
                replacement = reactive_atom_lines[reactive_index]
                output.append(replacement)
                reactive_index += 1
            else:
                output.append(line)
            atom_index += 1
        else:
            output.append(line)
    if reactive_index != reactive_atom_count:
        raise ValueError("Reactive atom replacement count did not match the requested reactive_atom_count.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc-master-pdb", required=True)
    parser.add_argument("--rc-reactive-pdb", required=True)
    parser.add_argument("--prod-reactive-pdb", required=True)
    parser.add_argument("--rc-output-pdb", required=True)
    parser.add_argument("--prod-output-pdb", required=True)
    parser.add_argument("--reactive-atom-count", required=True, type=int)
    args = parser.parse_args()

    master_lines = _read_pdb_lines(Path(args.rc_master_pdb))
    rc_reactive_lines = _extract_atom_records(_read_pdb_lines(Path(args.rc_reactive_pdb)))
    prod_reactive_lines = _extract_atom_records(_read_pdb_lines(Path(args.prod_reactive_pdb)))

    if len(rc_reactive_lines) < args.reactive_atom_count or len(prod_reactive_lines) < args.reactive_atom_count:
        raise ValueError("Reactive PDB inputs do not contain enough ATOM/HETATM lines.")

    rc_lines = _replace_reactive_region(master_lines, rc_reactive_lines[: args.reactive_atom_count], args.reactive_atom_count)
    prod_lines = _replace_reactive_region(master_lines, prod_reactive_lines[: args.reactive_atom_count], args.reactive_atom_count)

    Path(args.rc_output_pdb).write_text("\n".join(rc_lines) + "\n", encoding="utf-8")
    Path(args.prod_output_pdb).write_text("\n".join(prod_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
