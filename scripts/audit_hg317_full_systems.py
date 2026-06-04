from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from openmm import unit
from openmm.app import AmberPrmtopFile

from kemp_evb.openmm_backend import AmberSystemLoader, EVBSystemBuilder


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "prep" / "hg317_full_irc"
STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYX",
    "GLN",
    "GLU",
    "GLY",
    "HID",
    "HIE",
    "HIP",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "WAT",
    "HOH",
    "NA",
    "Na+",
    "CL",
    "Cl-",
}


def main() -> None:
    state1 = PREP / "state1_reactant.prmtop"
    coord1 = PREP / "state1_reactant.inpcrd"
    state2 = PREP / "state2_product.prmtop"
    coord2 = PREP / "state2_product.inpcrd"

    payload = {
        "state1": _summarize_prmtop(state1),
        "state2": _summarize_prmtop(state2),
        "plain_initial_energies_kj_mol": _read_minimization_energies(),
        "evb_compatible": False,
        "compatibility_error": None,
        "required_next_step": (
            "Build matched reactant/product diabatic topologies with identical atom count, atom order, masses, "
            "constraints, virtual sites, and periodic box. The current draft systems differ because the transferred "
            "proton is part of the substrate-like ligand in state 1 but not in the product-like GAFF2 ligand in state 2."
        ),
    }

    if state1.exists() and coord1.exists() and state2.exists() and coord2.exists():
        loader = AmberSystemLoader(nonbonded_method="PME", constraints="None")
        try:
            loaded1 = loader.load(str(state1), str(coord1))
            loaded2 = loader.load(str(state2), str(coord2))
            EVBSystemBuilder.validate_compatibility(loaded1, loaded2)
        except Exception as exc:
            payload["compatibility_error"] = str(exc)
        else:
            payload["evb_compatible"] = True

    out = PREP / "hg317_full_system_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _summarize_prmtop(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    prmtop = AmberPrmtopFile(str(path))
    atoms = list(prmtop.topology.atoms())
    residues = list(prmtop.topology.residues())
    system = prmtop.createSystem()
    masses = [system.getParticleMass(index).value_in_unit(unit.amu) for index in range(system.getNumParticles())]
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "atom_count": len(atoms),
        "residue_count": len(residues),
        "residue_counts": dict(Counter(residue.name for residue in residues)),
        "ligand_residues": [
            {"index": residue.index, "name": residue.name, "atom_count": len(list(residue.atoms()))}
            for residue in residues
            if residue.name not in STANDARD_RESIDUES
        ],
        "total_mass_amu": sum(masses),
    }


def _read_minimization_energies() -> dict:
    result = {}
    for state in ("state1", "state2"):
        path = PREP / "minimized" / state / f"{state}_minimization_summary.txt"
        if not path.exists():
            continue
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key.strip()] = float(value.strip())
        result[state] = values
    return result


if __name__ == "__main__":
    main()
