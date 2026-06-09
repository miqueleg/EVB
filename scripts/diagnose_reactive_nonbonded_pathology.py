#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from openmm.app import AmberPrmtopFile, PDBFile, PME


def _parse_pair(text: str) -> tuple[int, int]:
    left, right = text.replace(":", ",").split(",", 1)
    return int(left), int(right)


def _nonbonded_force(system):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise ValueError("System has no NonbondedForce.")


def _force_group_energies(prmtop_path: str, pdb_path: str, patch_pairs: list[tuple[int, int]] | None = None):
    prmtop = AmberPrmtopFile(prmtop_path)
    system = prmtop.createSystem(nonbondedMethod=PME, constraints=None)
    if patch_pairs:
        nonbonded = _nonbonded_force(system)
        existing = {frozenset(nonbonded.getExceptionParameters(index)[:2]) for index in range(nonbonded.getNumExceptions())}
        for atom1, atom2 in patch_pairs:
            if frozenset((atom1, atom2)) not in existing:
                nonbonded.addException(atom1, atom2, 0.0, 0.1, 0.0, replace=False)
    for index in range(system.getNumForces()):
        system.getForce(index).setForceGroup(index)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("CPU"))
    pdb = PDBFile(pdb_path)
    context.setPositions(pdb.positions)
    if prmtop.topology.getPeriodicBoxVectors() is not None:
        context.setPeriodicBoxVectors(*prmtop.topology.getPeriodicBoxVectors())
    total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    components = {}
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        energy = context.getState(getEnergy=True, groups={index}).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        components[type(force).__name__] = components.get(type(force).__name__, 0.0) + float(energy)
    return float(total), components


def _pair_nonbonded_terms(prmtop_path: str, pdb_path: str, pairs: list[tuple[int, int]]):
    prmtop = AmberPrmtopFile(prmtop_path)
    system = prmtop.createSystem(nonbondedMethod=PME, constraints=None)
    nonbonded = _nonbonded_force(system)
    positions = np.array([[vector.x, vector.y, vector.z] for vector in PDBFile(pdb_path).positions])
    rows = []
    for atom1, atom2 in pairs:
        q1, sigma1, epsilon1 = nonbonded.getParticleParameters(atom1)
        q2, sigma2, epsilon2 = nonbonded.getParticleParameters(atom2)
        exception = None
        for index in range(nonbonded.getNumExceptions()):
            ex_atom1, ex_atom2, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(index)
            if {ex_atom1, ex_atom2} == {atom1, atom2}:
                exception = (chargeprod, sigma, epsilon)
                break
        if exception is None:
            charge_product = q1.value_in_unit(unit.elementary_charge) * q2.value_in_unit(unit.elementary_charge)
            sigma_nm = 0.5 * (sigma1.value_in_unit(unit.nanometer) + sigma2.value_in_unit(unit.nanometer))
            epsilon_kj_mol = math.sqrt(
                epsilon1.value_in_unit(unit.kilojoule_per_mole) * epsilon2.value_in_unit(unit.kilojoule_per_mole)
            )
            excluded = False
        else:
            chargeprod, sigma, epsilon = exception
            charge_product = chargeprod.value_in_unit(unit.elementary_charge**2)
            sigma_nm = sigma.value_in_unit(unit.nanometer)
            epsilon_kj_mol = epsilon.value_in_unit(unit.kilojoule_per_mole)
            excluded = charge_product == 0.0 and epsilon_kj_mol == 0.0
        r_nm = float(np.linalg.norm(positions[atom1] - positions[atom2]))
        coul_kcal = 138.935456 * charge_product / r_nm / 4.184 if r_nm > 0.0 else float("inf")
        lj_kcal = 0.0
        if epsilon_kj_mol != 0.0 and r_nm > 0.0:
            lj_kcal = 4.0 * epsilon_kj_mol * ((sigma_nm / r_nm) ** 12 - (sigma_nm / r_nm) ** 6) / 4.184
        rows.append(
            {
                "atom1": atom1,
                "atom2": atom2,
                "distance_angstrom": r_nm * 10.0,
                "has_exception": exception is not None,
                "is_excluded": excluded,
                "coulomb_kcal_mol": coul_kcal,
                "lj_kcal_mol": lj_kcal,
                "pair_kcal_mol": coul_kcal + lj_kcal,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose EVB reactive-atom nonbonded singularities in two AMBER states.")
    parser.add_argument("--state1-prmtop", required=True)
    parser.add_argument("--state2-prmtop", required=True)
    parser.add_argument("--pdb", action="append", required=True, help="Label=path, may be repeated.")
    parser.add_argument("--pair", action="append", required=True, help="Atom pair as i,j. Indices are 0-based OpenMM indices.")
    parser.add_argument("--patch-state1-pair", action="append", default=[], help="State1 exception patch pair as i,j for controlled tests.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pdbs = dict(item.split("=", 1) for item in args.pdb)
    pairs = [_parse_pair(item) for item in args.pair]
    patch_pairs = [_parse_pair(item) for item in args.patch_state1_pair]
    report = {"pairs": pairs, "frames": {}}
    for label, pdb_path in pdbs.items():
        raw1, comp1 = _force_group_energies(args.state1_prmtop, pdb_path)
        raw2, comp2 = _force_group_energies(args.state2_prmtop, pdb_path)
        patched1, patched_comp1 = _force_group_energies(args.state1_prmtop, pdb_path, patch_pairs=patch_pairs)
        report["frames"][label] = {
            "state1_total_kcal_mol": raw1,
            "state2_total_kcal_mol": raw2,
            "state1_components_kcal_mol": comp1,
            "state2_components_kcal_mol": comp2,
            "state1_pair_terms": _pair_nonbonded_terms(args.state1_prmtop, pdb_path, pairs),
            "state2_pair_terms": _pair_nonbonded_terms(args.state2_prmtop, pdb_path, pairs),
            "state1_patched_total_kcal_mol": patched1,
            "state1_patched_components_kcal_mol": patched_comp1,
            "state1_patch_delta_kcal_mol": patched1 - raw1,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
