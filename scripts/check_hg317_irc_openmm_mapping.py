from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from kemp_evb.irc import identify_fixed_atoms, read_irc_xyz


ROOT = Path(__file__).resolve().parents[1]
IRC_XYZ = ROOT / "examples" / "HG3.17_CM_IRC.xyz"
PDB_5RGE = ROOT / "inputs" / "5RGE.pdb"
OPENMM_RC = ROOT / "prep" / "kemp_qm_openmm" / "07_openmm_bundles" / "RC" / "coordinates.pdb"
OPENMM_PROD = ROOT / "prep" / "kemp_qm_openmm" / "07_openmm_bundles" / "PROD" / "coordinates.pdb"
OUTPUT_DIR = ROOT / "outputs" / "hg317_cm_irc"


def main() -> None:
    frames = read_irc_xyz(IRC_XYZ)
    fixed_carbons = identify_fixed_atoms(frames, element="C")
    reference_atoms = _read_pdb_atoms(PDB_5RGE)
    rc_atoms = _read_pdb_atoms(OPENMM_RC)
    prod_atoms = _read_pdb_atoms(OPENMM_PROD)

    alpha_matches = _nearest_ca_matches(fixed_carbons, reference_atoms)
    irc_to_5rge = _direct_irc_to_reference_matches(frames, reference_atoms, tolerance_angstrom=0.08)
    ligand_matches = _nearest_ligand_matches(frames, reference_atoms, residue_name="6NT")
    report = {
        "status": "blocked",
        "reason": (
            "The IRC cluster is in the 5RGE crystal coordinate frame, but the current OpenMM EVB systems "
            "do not contain the HG3.17 protein atoms needed for IRC embedding and diabatic single-point evaluation."
        ),
        "irc": {
            "path": str(IRC_XYZ.relative_to(ROOT)),
            "n_frames": len(frames),
            "n_atoms": len(frames[0].symbols),
            "element_counts": dict(Counter(frames[0].symbols)),
            "fixed_carbon_count": len(fixed_carbons),
        },
        "reference_5rge": {
            "path": str(PDB_5RGE.relative_to(ROOT)),
            "n_atoms": len(reference_atoms),
            "residue_counts": dict(Counter(atom["residue_name"] for atom in reference_atoms)),
            "ca_count": sum(1 for atom in reference_atoms if atom["atom_name"] == "CA"),
        },
        "current_openmm_systems": {
            "state1_rc": _system_summary(OPENMM_RC, rc_atoms),
            "state2_prod": _system_summary(OPENMM_PROD, prod_atoms),
            "contains_protein_ca_atoms": any(atom["atom_name"] == "CA" for atom in rc_atoms + prod_atoms),
        },
        "alpha_carbon_anchor_matches": alpha_matches,
        "direct_irc_to_5rge_matches_0p08A": {
            "n_matched_atoms": len(irc_to_5rge),
            "matches": irc_to_5rge,
        },
        "nearest_5rge_6nt_ligand_matches": ligand_matches,
        "required_next_inputs": [
            "A full HG3.17 OpenMM diabatic state 1 system built from 5RGE that includes the protein/active-site atoms.",
            "A full HG3.17 OpenMM diabatic state 2 system with identical atom order, masses, constraints, virtual sites, and box.",
            "An IRC-to-OpenMM atom mapping for the reactive substrate/base/catalytic atoms, not only the CA anchors.",
            "A choice of whether to embed all 198 IRC cluster atoms or only the reactive subset with restrained protein relaxation.",
        ],
        "safe_workflow_after_inputs_exist": [
            "Insert each IRC frame into the full OpenMM state coordinates using the IRC-to-OpenMM atom mapping.",
            "Evaluate E1 and E2 single-point energies for every finite-energy IRC frame.",
            "Fit delta_alpha and H12 to the whole IRC profile.",
            "Plot QM vs fitted EVB, gap vs IRC frame, EVB weights, and residuals.",
            "Choose seed windows from IRC frames with smooth gap spacing.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "hg317_irc_openmm_mapping_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(OUTPUT_DIR / "hg317_irc_openmm_mapping_report.md", report)
    print(json.dumps({"status": report["status"], "reason": report["reason"], "report": str(OUTPUT_DIR / "hg317_irc_openmm_mapping_report.md")}, indent=2))


def _read_pdb_atoms(path: Path) -> list[dict]:
    atoms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[0]).upper()
        atoms.append(
            {
                "index": len(atoms),
                "serial": int(line[6:11]),
                "atom_name": atom_name,
                "residue_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "residue_number": line[22:26].strip(),
                "element": element,
                "coord_angstrom": np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
            }
        )
    return atoms


def _system_summary(path: Path, atoms: list[dict]) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "n_atoms": len(atoms),
        "residue_counts": dict(Counter(atom["residue_name"] for atom in atoms)),
        "ca_count": sum(1 for atom in atoms if atom["atom_name"] == "CA"),
    }


def _nearest_ca_matches(fixed_carbons, reference_atoms: list[dict]) -> list[dict]:
    ca_atoms = [atom for atom in reference_atoms if atom["atom_name"] == "CA"]
    rows = []
    for candidate in fixed_carbons:
        coord = np.asarray([candidate.x_angstrom, candidate.y_angstrom, candidate.z_angstrom], dtype=float)
        distance, atom = min(
            ((float(np.linalg.norm(coord - atom["coord_angstrom"])), atom) for atom in ca_atoms),
            key=lambda item: item[0],
        )
        rows.append(
            {
                "irc_atom_index": candidate.frame_atom_index,
                "pdb_atom_serial": atom["serial"],
                "chain_id": atom["chain_id"],
                "residue_name": atom["residue_name"],
                "residue_number": atom["residue_number"],
                "match_distance_angstrom": distance,
                "confidence": "exact" if distance < 0.05 else "review" if distance < 0.5 else "low",
            }
        )
    return rows


def _direct_irc_to_reference_matches(frames, reference_atoms: list[dict], tolerance_angstrom: float) -> list[dict]:
    coords = frames[0].coordinates_angstrom
    symbols = frames[0].symbols
    rows = []
    for irc_index, (symbol, coord) in enumerate(zip(symbols, coords)):
        candidates = [atom for atom in reference_atoms if atom["element"].upper() == symbol.upper()]
        if not candidates:
            continue
        distance, atom = min(
            ((float(np.linalg.norm(coord - atom["coord_angstrom"])), atom) for atom in candidates),
            key=lambda item: item[0],
        )
        if distance <= tolerance_angstrom:
            rows.append(
                {
                    "irc_atom_index": irc_index,
                    "element": symbol,
                    "pdb_atom_serial": atom["serial"],
                    "atom_name": atom["atom_name"],
                    "residue_name": atom["residue_name"],
                    "chain_id": atom["chain_id"],
                    "residue_number": atom["residue_number"],
                    "distance_angstrom": distance,
                }
            )
    return rows


def _nearest_ligand_matches(frames, reference_atoms: list[dict], residue_name: str) -> list[dict]:
    ligand_atoms = [atom for atom in reference_atoms if atom["residue_name"] == residue_name]
    coords = frames[0].coordinates_angstrom
    symbols = frames[0].symbols
    rows = []
    for atom in ligand_atoms:
        candidates = [(idx, sym, coords[idx]) for idx, sym in enumerate(symbols) if sym.upper() == atom["element"].upper()]
        distance, irc_index, symbol, coord = min(
            ((float(np.linalg.norm(coord - atom["coord_angstrom"])), idx, sym, coord) for idx, sym, coord in candidates),
            key=lambda item: item[0],
        )
        rows.append(
            {
                "pdb_atom_serial": atom["serial"],
                "pdb_atom_name": atom["atom_name"],
                "element": atom["element"],
                "nearest_irc_atom_index": irc_index,
                "distance_angstrom": distance,
            }
        )
    return rows


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# HG3.17 IRC to OpenMM Mapping Report",
        "",
        f"Status: **{report['status']}**",
        "",
        report["reason"],
        "",
        "## Current OpenMM Systems",
    ]
    for label, summary in report["current_openmm_systems"].items():
        if not isinstance(summary, dict):
            continue
        lines.append(f"- {label}: {summary['n_atoms']} atoms, residues {summary['residue_counts']}, CA atoms {summary['ca_count']}")
    lines.extend(
        [
            "",
            "## Alpha-Carbon Anchors",
            "",
            "| IRC atom | 5RGE residue | serial | distance A | confidence |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in report["alpha_carbon_anchor_matches"]:
        lines.append(
            f"| {row['irc_atom_index']} | {row['residue_name']} {row['chain_id']}{row['residue_number']} | "
            f"{row['pdb_atom_serial']} | {row['match_distance_angstrom']:.4f} | {row['confidence']} |"
        )
    lines.extend(["", "## Required Next Inputs", ""])
    lines.extend(f"- {item}" for item in report["required_next_inputs"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
