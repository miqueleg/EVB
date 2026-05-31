from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdb", required=True)
    parser.add_argument("--output-pdb", required=True)
    parser.add_argument("--reactive-atom-count", required=True, type=int)
    args = parser.parse_args()

    input_path = Path(args.input_pdb)
    output_path = Path(args.output_pdb)
    lines = input_path.read_text(encoding="utf-8").splitlines()

    out: list[str] = []
    seen_atoms = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            seen_atoms += 1
            if seen_atoms <= args.reactive_atom_count:
                continue
        out.append(line)

    output_path.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
