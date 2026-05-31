from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from kemp_evb.tools import sync_atom_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize atom names across two EVB states by assigning shared unique names.")
    parser.add_argument("--state1-prmtop", required=True)
    parser.add_argument("--state1-pdb", required=True)
    parser.add_argument("--state2-prmtop", required=True)
    parser.add_argument("--state2-pdb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="E", help="Prefix for generated unique atom names, default: E")
    args = parser.parse_args()

    result = sync_atom_names(
        state1_prmtop=args.state1_prmtop,
        state1_pdb=args.state1_pdb,
        state2_prmtop=args.state2_prmtop,
        state2_pdb=args.state2_pdb,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
