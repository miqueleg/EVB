from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from kemp_evb.tools import sync_shared_box


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize two EVB states onto one shared periodic box by rewriting both PDB CRYST1 and prmtop BOX_DIMENSIONS.")
    parser.add_argument("--state1-prmtop", required=True)
    parser.add_argument("--state1-pdb", required=True)
    parser.add_argument("--state2-prmtop", required=True)
    parser.add_argument("--state2-pdb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["state1", "state2", "average"], default="state1", help="Choose which box to copy onto both states.")
    parser.add_argument("--wrap-coordinates", action="store_true", help="Wrap PDB coordinates back into the chosen box.")
    args = parser.parse_args()

    result = sync_shared_box(
        state1_prmtop=args.state1_prmtop,
        state1_pdb=args.state1_pdb,
        state2_prmtop=args.state2_prmtop,
        state2_pdb=args.state2_pdb,
        output_dir=args.output_dir,
        source=args.source,
        wrap_coordinates=args.wrap_coordinates,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
