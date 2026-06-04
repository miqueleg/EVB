from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "prep" / "hg317_full_irc"


def main() -> None:
    _write_state1_solvent_template()
    _write_patched_product_ligand()
    _write_product_asp_h197_frcmod()
    _write_tleap_inputs()
    payload = {
        "status": "prepared_matched_draft_inputs",
        "warning": (
            "These synchronized draft states reuse the exact state-1 solvent/ion template and a 16-atom product ligand. "
            "State 2 bonds H197 to ASP127 OD2 with minimal matched-topology parameters. Those reactive-region terms "
            "are placeholders and should be replaced by parameters validated against the QM model before publication use."
        ),
        "state1_reference": "prep/hg317_full_irc/state1_reactant.prmtop",
        "state2_matched_tleap": "prep/hg317_full_irc/tleap_state2_matched16.in",
    }
    (PREP / "matched_draft_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _write_state1_solvent_template() -> None:
    source = PREP / "state1_reactant_solvated.pdb"
    if not source.exists():
        raise FileNotFoundError(source)
    lines = []
    seen_water_residues: set[tuple[str, str, str]] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue = line[17:20].strip()
        atom = line[12:16].strip()
        if residue in {"HOH", "WAT"}:
            water_key = (line[21].strip(), line[22:26].strip(), line[26].strip())
            seen_water_residues.add(water_key)
            if len(seen_water_residues) <= 6:
                continue
            lines.append(line)
        elif atom == "Na+" or residue in {"Na+", "NA", "Cl-", "CL"}:
            lines.append(line)
    if not lines:
        raise ValueError("No solvent/ion records found in state1_reactant_solvated.pdb.")
    (PREP / "state1_solvent_ions_template.pdb").write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")


def _write_patched_product_ligand() -> None:
    source = PREP / "state2_product_ligand_matched16.mol2"
    if not source.exists():
        raise FileNotFoundError(source)
    lines = source.read_text(encoding="utf-8").splitlines()
    patched = []
    for line in lines:
        if line.split()[:2] == ["16", "H4"]:
            line = line.replace(" DU ", " ho ")
        patched.append(line)
    (PREP / "state2_product_ligand_matched16_patched.mol2").write_text("\n".join(patched) + "\n", encoding="utf-8")


def _write_product_asp_h197_frcmod() -> None:
    (PREP / "state2_asp127_h197.frcmod").write_text(
        """Matched EVB product-state ASP127-OD2--H197 parameters
MASS

BOND
O2-ho  553.0  0.960  adapted from hydroxyl O-H for matched EVB ASP127-H197 topology

ANGLE
CO-O2-ho   50.0  113.00  adapted from carboxylic-acid C-O-H angle

DIHE
2C-CO-O2-ho   1    0.000      0.0    3.0  reactive-region matched topology placeholder
O2-CO-O2-ho   1    0.000    180.0    2.0  reactive-region matched topology placeholder

IMPROPER

NONBON
""",
        encoding="utf-8",
    )


def _write_tleap_inputs() -> None:
    (PREP / "tleap_state1_matched16.in").write_text(
        """source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams state1_reactant_ligand.frcmod
lig = loadmol2 state1_reactant_ligand.mol2
prot = loadpdb 5RGE_clean_active_site_waters_noH.pdb
solv = loadpdb state1_solvent_ions_template.pdb
model = combine {prot lig solv}
check model
set model box {85.234868 65.595866 73.747658}
saveamberparm model state1_reactant_matched16.prmtop state1_reactant_matched16.inpcrd
savepdb model state1_reactant_matched16.pdb
quit
""",
        encoding="utf-8",
    )
    (PREP / "tleap_state2_matched16.in").write_text(
        """source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams state2_product_ligand.frcmod
loadamberparams state2_asp127_h197.frcmod
lig = loadmol2 state2_product_ligand_matched16_patched.mol2
prot = loadpdb 5RGE_clean_active_site_waters_noH.pdb
solv = loadpdb state1_solvent_ions_template.pdb
model = combine {prot lig solv}
bond model.127.OD2 model.310.H4
check model
set model box {85.234868 65.595866 73.747658}
saveamberparm model state2_product_matched16.prmtop state2_product_matched16.inpcrd
savepdb model state2_product_matched16.pdb
quit
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
