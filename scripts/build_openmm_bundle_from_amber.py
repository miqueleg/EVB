from __future__ import annotations

from pathlib import Path

from kemp_evb.openmm_backend import AmberSystemLoader, write_openmm_bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create an OpenMM-native state bundle from Amber topology plus coordinate file.")
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--coordinates", required=True, help="PDB or Amber restart/inpcrd file with the desired coordinates.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nonbonded-method", default="PME")
    parser.add_argument("--constraints", default="None")
    args = parser.parse_args()

    loader = AmberSystemLoader(nonbonded_method=args.nonbonded_method, constraints=args.constraints)
    state = loader.load(args.prmtop, args.coordinates)
    write_openmm_bundle(
        Path(args.output_dir),
        system=state.system,
        topology=state.topology,
        positions_nm=state.positions_nm,
        box_vectors_nm=state.box_vectors_nm,
    )


if __name__ == "__main__":
    main()
