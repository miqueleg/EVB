from __future__ import annotations

import argparse
from pathlib import Path


COMMON_ATOM_NAMES = [f"E{i:02d}" for i in range(1, 24)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resname", default="EVB")
    parser.add_argument("--names", help="Comma-separated atom names to apply in order.")
    parser.add_argument("--names-file", help="Text file containing one atom name per line.")
    args = parser.parse_args()

    if args.names and args.names_file:
        raise ValueError("Use only one of --names or --names-file.")
    if args.names_file:
        atom_names = [line.strip() for line in Path(args.names_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.names:
        atom_names = [item.strip() for item in args.names.split(",") if item.strip()]
    else:
        atom_names = COMMON_ATOM_NAMES

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    in_atom = False
    atom_count = 0
    out_lines: list[str] = []
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            atom_count = 0
            out_lines.append(line)
            continue
        if line.startswith("@<TRIPOS>") and not line.startswith("@<TRIPOS>ATOM"):
            in_atom = False
            out_lines.append(line)
            continue
        if in_atom and line.strip():
            atom_count += 1
            fields = line.split()
            if atom_count > len(atom_names):
                raise ValueError(f"Unexpected atom count in mol2: {atom_count}")
            fields[1] = atom_names[atom_count - 1]
            if len(fields) >= 8:
                fields[7] = args.resname
            out_lines.append(
                f"{int(fields[0]):7d} {fields[1]:<4s} {float(fields[2]):10.4f} {float(fields[3]):10.4f} {float(fields[4]):10.4f} "
                f"{fields[5]:<8s} {int(fields[6]):4d} {fields[7]:<8s} {float(fields[8]):10.6f}"
            )
        else:
            out_lines.append(line)
    Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
