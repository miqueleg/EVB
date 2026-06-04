from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kemp_evb.irc import identify_fixed_atoms, read_irc_xyz


ROOT = Path(__file__).resolve().parents[1]
IRC_XYZ = ROOT / "examples" / "HG3.17_CM_IRC.xyz"
PDB_5RGE = ROOT / "inputs" / "5RGE.pdb"
OUTPUT_DIR = ROOT / "outputs" / "hg317_cm_irc"


def main() -> None:
    frames = read_irc_xyz(IRC_XYZ)
    fixed_carbons = identify_fixed_atoms(frames, element="C")
    ca_atoms = _read_ca_atoms(PDB_5RGE)
    mappings = []
    for candidate in fixed_carbons:
        coord = np.asarray([candidate.x_angstrom, candidate.y_angstrom, candidate.z_angstrom], dtype=float)
        distances = sorted(
            (float(np.linalg.norm(coord - atom["coord_angstrom"])), atom)
            for atom in ca_atoms
        )
        distance, atom = distances[0]
        mappings.append(
            {
                "irc_atom_index": candidate.frame_atom_index,
                "irc_element": candidate.element,
                "irc_coordinate_angstrom": [
                    candidate.x_angstrom,
                    candidate.y_angstrom,
                    candidate.z_angstrom,
                ],
                "pdb_id": "5RGE",
                "pdb_atom_serial": atom["serial"],
                "pdb_atom_name": "CA",
                "chain_id": atom["chain_id"],
                "residue_name": atom["residue_name"],
                "residue_number": atom["residue_number"],
                "pdb_coordinate_angstrom": atom["coord_angstrom"].tolist(),
                "match_distance_angstrom": distance,
                "confidence": "exact" if distance < 0.05 else "review" if distance < 0.5 else "low",
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_yaml(OUTPUT_DIR / "hg317_5rge_irc_ca_mapping.yaml", mappings)
    (OUTPUT_DIR / "hg317_5rge_irc_ca_mapping.json").write_text(json.dumps({"alpha_carbon_mapping": mappings}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(mappings)} fixed-carbon to 5RGE CA mappings into {OUTPUT_DIR}")


def _read_ca_atoms(path: Path) -> list[dict]:
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        if atom_name != "CA" or altloc not in ("", "A"):
            continue
        atoms.append(
            {
                "serial": int(line[6:11]),
                "residue_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "residue_number": int(line[22:26]),
                "coord_angstrom": np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
            }
        )
    if not atoms:
        raise ValueError(f"No CA atoms found in {path}.")
    return atoms


def _write_yaml(path: Path, mappings: list[dict]) -> None:
    lines = [
        "# HG3.17 cluster-model IRC fixed-carbon anchors mapped to 5RGE alpha carbons.",
        "# 5RGE source: RCSB PDB entry 5RGE, HG3.17 transition-state-analog crystal structure.",
        "# Distances are direct coordinate differences in Angstrom because the fixed IRC carbons",
        "# are in the 5RGE crystal coordinate frame.",
        "reference_pdb: inputs/5RGE.pdb",
        "irc_xyz: examples/HG3.17_CM_IRC.xyz",
        "alpha_carbon_mapping:",
    ]
    for mapping in mappings:
        lines.extend(
            [
                f"  - irc_atom_index: {mapping['irc_atom_index']}",
                f"    irc_element: {mapping['irc_element']}",
                "    irc_coordinate_angstrom: "
                f"[{mapping['irc_coordinate_angstrom'][0]:.6f}, "
                f"{mapping['irc_coordinate_angstrom'][1]:.6f}, "
                f"{mapping['irc_coordinate_angstrom'][2]:.6f}]",
                "    pdb:",
                f"      id: {mapping['pdb_id']}",
                f"      atom_serial: {mapping['pdb_atom_serial']}",
                f"      atom_name: {mapping['pdb_atom_name']}",
                f"      chain_id: {mapping['chain_id']}",
                f"      residue_name: {mapping['residue_name']}",
                f"      residue_number: {mapping['residue_number']}",
                "      coordinate_angstrom: "
                f"[{mapping['pdb_coordinate_angstrom'][0]:.6f}, "
                f"{mapping['pdb_coordinate_angstrom'][1]:.6f}, "
                f"{mapping['pdb_coordinate_angstrom'][2]:.6f}]",
                f"    match_distance_angstrom: {mapping['match_distance_angstrom']:.6f}",
                f"    confidence: {mapping['confidence']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
