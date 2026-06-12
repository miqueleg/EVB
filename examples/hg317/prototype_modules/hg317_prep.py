from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .irc import read_irc_xyz

DEFAULT_REACTIVE_COMPONENT_ATOMS = [6, 7, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 194, 195, 196, 197]
DEFAULT_PRODUCT_LIGAND_ATOMS = [6, 7, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 194, 195, 196]
REQUIRED_AMBERTOOLS = ["antechamber", "parmchk2", "tleap"]


def prepare_hg317_system(config_path: str | Path, execute: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("HG3.17 preparation config must be a mapping.")
    output_dir = Path(config.get("output_dir", "prep/hg317_system_prep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pdb = Path(config.get("source_pdb", "inputs/5RGE.pdb"))
    irc_xyz = Path(config.get("irc_xyz", "examples/HD3.17_IRC.xyz"))
    if not source_pdb.exists():
        raise FileNotFoundError(f"Source PDB not found: {source_pdb}")
    if not irc_xyz.exists():
        raise FileNotFoundError(f"IRC XYZ not found: {irc_xyz}")
    missing_tools = [tool for tool in REQUIRED_AMBERTOOLS if shutil.which(tool) is None]
    if execute and missing_tools:
        raise RuntimeError("AmberTools executables are required but missing: " + ", ".join(missing_tools))

    ligand_residue = config.get("ligand_residue_name", "6NT")
    water_cutoff = float(config.get("active_site_water_cutoff_angstrom", 6.0))
    product_frame = int(config.get("product_frame", 0))
    reactant_frame = int(config.get("reactant_frame", 106))
    reactive_atoms = list(config.get("reactive_component_atoms", DEFAULT_REACTIVE_COMPONENT_ATOMS))
    product_atoms = list(config.get("product_ligand_atoms", DEFAULT_PRODUCT_LIGAND_ATOMS))
    frames = read_irc_xyz(str(irc_xyz))
    records = _read_pdb_records(source_pdb)
    ligand_records = [record for record in records if record["residue_name"] == ligand_residue]
    if not ligand_records:
        raise ValueError(f"No ligand residue {ligand_residue!r} found in {source_pdb}")
    ligand_coords = np.asarray([record["coord"] for record in ligand_records])
    cleaned, retained_waters = _clean_records(records, ligand_coords, ligand_residue, water_cutoff, config.get("active_site_waters", {}))
    cleaned_heavy = [record for record in cleaned if record["element"] != "H"]
    _write_pdb(output_dir / "5RGE_clean_active_site_waters.pdb", cleaned)
    _write_pdb(output_dir / "5RGE_clean_active_site_waters_noH.pdb", cleaned_heavy)
    _write_ligand_pdb(output_dir / "irc_reactant_ligand_state1.pdb", frames[reactant_frame], reactive_atoms, "SBR")
    _write_ligand_pdb(output_dir / "irc_product_ligand_state2.pdb", frames[product_frame], product_atoms, "SDP")
    _write_ligand_pdb(output_dir / "irc_product_ligand_state2_matched.pdb", frames[product_frame], reactive_atoms, "SDM")
    _write_driver_scripts(output_dir, int(config.get("state1_ligand_charge", 0)), int(config.get("state2_ligand_charge", 0)))
    report = {
        "status": "prepared_intermediates" if not execute else "ambertools_executed",
        "source_pdb": str(source_pdb),
        "irc_xyz": str(irc_xyz),
        "output_dir": str(output_dir),
        "ambertools_required": REQUIRED_AMBERTOOLS,
        "ambertools_missing": missing_tools,
        "execute": execute,
        "protein_force_field": "ff14SB",
        "ligand_force_field": "GAFF2",
        "charge_model": "AM1-BCC via antechamber",
        "protonation_choices": dict(config.get("protonation", {})),
        "user_supplied_ligands": dict(config.get("user_supplied_ligands", {})),
        "retained_waters": retained_waters,
        "active_site_water_cutoff_angstrom": water_cutoff,
        "removed_bound_ligand_atoms": len(ligand_records),
        "cleaned_atom_count": len(cleaned),
        "cleaned_heavy_atom_count": len(cleaned_heavy),
        "cleaned_residue_counts": dict(Counter(record["residue_name"] for record in cleaned)),
        "scientific_caution": "This scaffold writes transparent intermediates and AmberTools drivers. HG3.17 is not claimed reproducible from a fresh clone until the transferred-proton topology is validated with identical state atom order and masses.",
        "next_steps": ["review retained waters/protonation", "run AmberTools when available", "validate matched prmtops", "run evb prepare-evb-inputs and setup-from-irc"],
    }
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if execute:
        subprocess.run(["bash", "run_ambertools.sh"], cwd=output_dir, check=True)
    return report


def _read_pdb_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        records.append({"record_name": line[:6].strip(), "atom_name": line[12:16].strip(), "residue_name": line[17:20].strip(), "chain_id": line[21].strip() or "A", "residue_number": int(line[22:26]), "insertion_code": line[26].strip(), "coord": np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])]), "occupancy": float(line[54:60]), "bfactor": float(line[60:66]), "element": (line[76:78].strip() or line[12:16].strip()[0]).upper()})
    return records


def _clean_records(records, ligand_coords, ligand_residue, water_cutoff, water_rules):
    explicit_keep = {tuple(item) for item in water_rules.get("retain_residues", [])}
    retained = []
    cleaned = []
    for record in records:
        if record["residue_name"] == ligand_residue:
            continue
        if record["residue_name"] == "HOH":
            key = (record["chain_id"], record["residue_number"])
            distance = float(np.min(np.linalg.norm(ligand_coords - record["coord"], axis=1)))
            if distance > water_cutoff and key not in explicit_keep:
                continue
            if key not in retained:
                retained.append(key)
        cleaned.append(record)
    return cleaned, [{"chain": chain, "residue_number": number} for chain, number in retained]


def _write_pdb(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for serial, record in enumerate(records, start=1):
            x, y, z = record["coord"]
            handle.write(f"{record['record_name']:<6s}{serial:5d} {record['atom_name']:<4s} {record['residue_name']:>3s} {record['chain_id']}{record['residue_number']:4d}{record['insertion_code'] or ' '}   {x:8.3f}{y:8.3f}{z:8.3f}{record['occupancy']:6.2f}{record['bfactor']:6.2f}          {record['element']:>2s}\n")
        handle.write("END\n")


def _write_ligand_pdb(path: Path, frame, atom_indices: list[int], residue_name: str) -> None:
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for serial, atom_index in enumerate(atom_indices, start=1):
            symbol = frame.symbols[atom_index]
            counts[symbol] += 1
            x, y, z = frame.coordinates_angstrom[atom_index]
            handle.write(f"HETATM{serial:5d} {symbol + str(counts[symbol]):<4s} {residue_name:>3s} B   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol:>2s}\n")
        handle.write("END\n")


def _write_driver_scripts(output_dir: Path, charge1: int, charge2: int) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "for exe in antechamber parmchk2 tleap; do",
        "  command -v \"$exe\" >/dev/null || { echo \"Missing AmberTools executable: $exe\" >&2; exit 1; }",
        "done",
        "",
        f"antechamber -i irc_reactant_ligand_state1.pdb -fi pdb -o state1_reactant_ligand.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc {charge1}",
        "parmchk2 -i state1_reactant_ligand.mol2 -f mol2 -o state1_reactant_ligand.frcmod -s gaff2",
        f"antechamber -i irc_product_ligand_state2_matched.pdb -fi pdb -o state2_product_ligand_matched.mol2 -fo mol2 -at gaff2 -c bcc -s 2 -nc {charge2}",
        "parmchk2 -i state2_product_ligand_matched.mol2 -f mol2 -o state2_product_ligand_matched.frcmod -s gaff2",
        "",
        "tleap -f tleap_state1.in",
        "tleap -f tleap_state2.in",
    ]
    (output_dir / "run_ambertools.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state1 = ["source leaprc.protein.ff14SB", "source leaprc.gaff2", "source leaprc.water.tip3p", "loadamberparams state1_reactant_ligand.frcmod", "lig = loadmol2 state1_reactant_ligand.mol2", "prot = loadpdb 5RGE_clean_active_site_waters_noH.pdb", "model = combine {prot lig}", "check model", "solvatebox model TIP3PBOX 10.0", "addions model Na+ 0", "addions model Cl- 0", "saveamberparm model state1_reactant.prmtop state1_reactant.inpcrd", "savepdb model state1_reactant_solvated.pdb", "quit"]
    state2 = ["source leaprc.protein.ff14SB", "source leaprc.gaff2", "source leaprc.water.tip3p", "loadamberparams state2_product_ligand_matched.frcmod", "lig = loadmol2 state2_product_ligand_matched.mol2", "prot = loadpdb 5RGE_clean_active_site_waters_noH.pdb", "model = combine {prot lig}", "check model", "solvatebox model TIP3PBOX 10.0", "addions model Na+ 0", "addions model Cl- 0", "saveamberparm model state2_product.prmtop state2_product.inpcrd", "savepdb model state2_product_solvated.pdb", "quit"]
    (output_dir / "tleap_state1.in").write_text("\n".join(state1) + "\n", encoding="utf-8")
    (output_dir / "tleap_state2.in").write_text("\n".join(state2) + "\n", encoding="utf-8")
