from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from kemp_evb.irc import read_irc_xyz


ROOT = Path(__file__).resolve().parents[1]
PDB_5RGE = ROOT / "inputs" / "5RGE.pdb"
IRC_XYZ = ROOT / "examples" / "HG3.17_CM_IRC.xyz"
OUT = ROOT / "prep" / "hg317_full_irc"

PRODUCT_FRAME = 0
REACTANT_FRAME = 106
TS_FRAME = 41
REACTIVE_COMPONENT_ATOMS = [6, 7, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 194, 195, 196, 197]
PRODUCT_LIGAND_ATOMS = [6, 7, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 194, 195, 196]
PRODUCT_MATCHED_LIGAND_ATOMS = REACTIVE_COMPONENT_ATOMS
REACTANT_LIGAND_ATOMS = REACTIVE_COMPONENT_ATOMS
ACTIVE_SITE_WATER_CUTOFF_ANGSTROM = 6.0


def main() -> None:
    frames = read_irc_xyz(IRC_XYZ)
    product = frames[PRODUCT_FRAME]
    reactant = frames[REACTANT_FRAME]
    OUT.mkdir(parents=True, exist_ok=True)

    records = _read_pdb_records(PDB_5RGE)
    ligand_records = [record for record in records if record["residue_name"] == "6NT"]
    ligand_coords = np.asarray([record["coord"] for record in ligand_records], dtype=float)
    cleaned_records = _clean_5rge(records, ligand_coords)
    cleaned_heavy_records = _strip_hydrogens(cleaned_records)
    _write_pdb(OUT / "5RGE_clean_active_site_waters.pdb", cleaned_records)
    _write_pdb(OUT / "5RGE_clean_active_site_waters_noH.pdb", cleaned_heavy_records)
    _write_pdb(OUT / "5RGE_clean_no_ligand_no_bulk_water.pdb", [record for record in cleaned_records if record["residue_name"] != "HOH"])

    _write_ligand_pdb(
        OUT / "irc_reactant_ligand_state1.pdb",
        reactant,
        REACTANT_LIGAND_ATOMS,
        residue_name="SBR",
        comment="IRC frame 106, reactant-like substrate, includes transferred proton H197.",
    )
    _write_ligand_pdb(
        OUT / "irc_product_ligand_state2.pdb",
        product,
        PRODUCT_LIGAND_ATOMS,
        residue_name="SDP",
        comment="IRC frame 0, product-like ligand, transferred proton is on protein fragment in the IRC.",
    )
    _write_ligand_pdb(
        OUT / "irc_product_ligand_state2_matched16.pdb",
        product,
        PRODUCT_MATCHED_LIGAND_ATOMS,
        residue_name="SDM",
        comment="IRC frame 0, product-like matched 16-atom ligand for draft atom-count synchronization.",
    )
    _write_ligand_pdb(
        OUT / "irc_product_transferred_proton_H197.pdb",
        product,
        [197],
        residue_name="HPT",
        comment="Transferred proton in product-like IRC frame; should belong to protonated catalytic base topology.",
    )
    _write_combined_pdb(OUT / "state1_reactant_model_unsolvated.pdb", cleaned_records, OUT / "irc_reactant_ligand_state1.pdb")
    _write_combined_pdb(OUT / "state2_product_model_unsolvated_incomplete.pdb", cleaned_records, OUT / "irc_product_ligand_state2.pdb")
    _write_ambertools_driver()
    _write_report(cleaned_records, cleaned_heavy_records, ligand_records)
    print(json.dumps({"output_dir": str(OUT), "status": "prepared_inputs_not_final_evb_systems"}, indent=2))


def _read_pdb_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        records.append(
            {
                "record_name": line[:6].strip(),
                "serial": int(line[6:11]),
                "atom_name": line[12:16].strip(),
                "residue_name": line[17:20].strip(),
                "chain_id": line[21].strip() or "A",
                "residue_number": int(line[22:26]),
                "insertion_code": line[26].strip(),
                "coord": np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
                "occupancy": float(line[54:60]),
                "bfactor": float(line[60:66]),
                "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
            }
        )
    return records


def _clean_5rge(records: list[dict], ligand_coords: np.ndarray) -> list[dict]:
    cleaned = []
    for record in records:
        if record["residue_name"] == "6NT":
            continue
        if record["residue_name"] == "HOH":
            distance = float(np.min(np.linalg.norm(ligand_coords - record["coord"], axis=1)))
            if distance > ACTIVE_SITE_WATER_CUTOFF_ANGSTROM:
                continue
        cleaned.append(record)
    return cleaned


def _strip_hydrogens(records: list[dict]) -> list[dict]:
    return [record for record in records if record["element"] != "H"]


def _write_pdb(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        serial = 1
        for record in records:
            x, y, z = record["coord"]
            handle.write(
                f"{record['record_name']:<6s}{serial:5d} {record['atom_name']:<4s} "
                f"{record['residue_name']:>3s} {record['chain_id']}{record['residue_number']:4d}{record['insertion_code'] or ' '}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{record['occupancy']:6.2f}{record['bfactor']:6.2f}          {record['element']:>2s}\n"
            )
            serial += 1
        handle.write("END\n")


def _write_ligand_pdb(path: Path, frame, atom_indices: list[int], residue_name: str, comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"REMARK {comment}\n")
        counts: Counter[str] = Counter()
        for serial, atom_index in enumerate(atom_indices, start=1):
            symbol = frame.symbols[atom_index]
            counts[symbol] += 1
            atom_name = f"{symbol}{counts[symbol]}"
            x, y, z = frame.coordinates_angstrom[atom_index]
            handle.write(
                f"HETATM{serial:5d} {atom_name:<4s} {residue_name:>3s} B   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


def _write_combined_pdb(path: Path, protein_records: list[dict], ligand_pdb: Path) -> None:
    ligand_lines = [line for line in ligand_pdb.read_text(encoding="utf-8").splitlines() if line.startswith("HETATM")]
    with path.open("w", encoding="utf-8") as handle:
        serial = 1
        for record in protein_records:
            x, y, z = record["coord"]
            handle.write(
                f"{record['record_name']:<6s}{serial:5d} {record['atom_name']:<4s} "
                f"{record['residue_name']:>3s} {record['chain_id']}{record['residue_number']:4d}{record['insertion_code'] or ' '}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{record['occupancy']:6.2f}{record['bfactor']:6.2f}          {record['element']:>2s}\n"
            )
            serial += 1
        for line in ligand_lines:
            handle.write(f"HETATM{serial:5d}{line[11:]}\n")
            serial += 1
        handle.write("END\n")


def _write_ambertools_driver() -> None:
    (OUT / "README_preparation.md").write_text(
        """# HG3.17 Full-System IRC Preparation

This directory contains cleaned 5RGE-derived model inputs and IRC-derived ligand seeds.

Important: these are not final EVB E1/E2 systems yet. The reactant and product states differ by a proton transfer between the ligand and the catalytic protein fragment. To make a valid two-state EVB system, both states must contain the same atoms in the same order and masses, with different bonding/charges only in the diabatic topologies.

The current generated model is enough to start AmberTools preparation, but the transferred proton H197 needs a custom topology decision:

- state 1/reactant: H197 is on the substrate-like ligand.
- state 2/product: H197 is on the catalytic-base/protein fragment.

Standard ff14SB + GAFF2 cannot guarantee identical atom ordering across that proton-transfer topology unless we create matched custom residues or manually reorder/validate the prmtops.
""",
        encoding="utf-8",
    )
    (OUT / "run_ambertools_draft.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"
source /home/mestevez/Programs/Amber24_AT25/ambertools25/amber.sh

cd "$(dirname "$0")"

# Draft ligand parameterization. Charges must be reviewed for the actual Kemp states.
antechamber -i irc_reactant_ligand_state1.pdb -fi pdb -o state1_reactant_ligand.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc 0
parmchk2 -i state1_reactant_ligand.mol2 -f mol2 -o state1_reactant_ligand.frcmod -s gaff2

antechamber -i irc_product_ligand_state2.pdb -fi pdb -o state2_product_ligand.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc -1
parmchk2 -i state2_product_ligand.mol2 -f mol2 -o state2_product_ligand.frcmod -s gaff2

antechamber -i irc_product_ligand_state2_matched16.pdb -fi pdb -o state2_product_ligand_matched16.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc 0
parmchk2 -i state2_product_ligand_matched16.mol2 -f mol2 -o state2_product_ligand_matched16.frcmod -s gaff2

echo "Draft GAFF2/BCC ligand files generated. Do not run EVB until transferred-proton topology is resolved and state compatibility is validated."
""",
        encoding="utf-8",
    )
    (OUT / "tleap_state1_draft.in").write_text(
        """source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams state1_reactant_ligand.frcmod
lig = loadmol2 state1_reactant_ligand.mol2
prot = loadpdb 5RGE_clean_active_site_waters_noH.pdb
model = combine {prot lig}
check model
solvatebox model TIP3PBOX 10.0
addions model Na+ 0
addions model Cl- 0
saveamberparm model state1_reactant.prmtop state1_reactant.inpcrd
savepdb model state1_reactant_solvated.pdb
quit
""",
        encoding="utf-8",
    )
    (OUT / "tleap_state2_draft.in").write_text(
        """source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams state2_product_ligand.frcmod
lig = loadmol2 state2_product_ligand.mol2
prot = loadpdb 5RGE_clean_active_site_waters_noH.pdb
model = combine {prot lig}
check model
solvatebox model TIP3PBOX 10.0
addions model Na+ 0
addions model Cl- 0
saveamberparm model state2_product.prmtop state2_product.inpcrd
savepdb model state2_product_solvated.pdb
quit
""",
        encoding="utf-8",
    )


def _write_report(cleaned_records: list[dict], cleaned_heavy_records: list[dict], ligand_records: list[dict]) -> None:
    payload = {
        "status": "prepared_inputs_not_final_evb_systems",
        "source_pdb": str(PDB_5RGE.relative_to(ROOT)),
        "irc_xyz": str(IRC_XYZ.relative_to(ROOT)),
        "active_site_water_cutoff_angstrom": ACTIVE_SITE_WATER_CUTOFF_ANGSTROM,
        "removed_residue": "6NT",
        "removed_bound_ligand_atoms": len(ligand_records),
        "cleaned_atom_count": len(cleaned_records),
        "cleaned_heavy_atom_count": len(cleaned_heavy_records),
        "cleaned_residue_counts": dict(Counter(record["residue_name"] for record in cleaned_records)),
        "reactant_frame": REACTANT_FRAME,
        "product_frame": PRODUCT_FRAME,
        "ts_frame_without_energy": TS_FRAME,
        "reactant_ligand_irc_atoms": REACTANT_LIGAND_ATOMS,
        "product_ligand_irc_atoms": PRODUCT_LIGAND_ATOMS,
        "product_matched_ligand_irc_atoms": PRODUCT_MATCHED_LIGAND_ATOMS,
        "transferred_proton_irc_atom": 197,
        "blocking_issue": (
            "The transferred proton belongs to the substrate in the reactant and to the catalytic protein fragment in the product. "
            "A valid EVB pair requires matched atom order and masses across both prmtops, likely via custom matched residues or manual prmtop reordering."
        ),
    }
    (OUT / "preparation_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
