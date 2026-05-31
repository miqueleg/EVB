from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import openmm as mm
from openmm import app, unit
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from kemp_evb.openmm_backend import EVBSystemBuilder, LoadedAmberState, write_pdb
from kemp_evb.simulation import EVBSimulation, create_integrator
from kemp_evb.solution_reference import PRODUCT_COMPLEX_SMILES, REACTANT_COMPLEX_SMILES, REFERENCE_TARGETS

TRANSFER_MAP = 17
NONTRANSFER_H_LABELS = [
    (5, 1),
    (6, 1),
    (12, 1),
    (13, 1),
    (13, 2),
    (13, 3),
]
HEAVY_MAPS = list(range(1, 17))


def main() -> None:
    output_dir = Path("outputs/solution_reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    reactant_rdmol = build_complex_rdmol(REACTANT_COMPLEX_SMILES, transfer_host_map=10)
    product_rdmol = build_complex_rdmol(PRODUCT_COMPLEX_SMILES, transfer_host_map=16)
    reactant_rdmol = place_acetate_near_reactive_center(reactant_rdmol)
    product_rdmol = align_to_reference(product_rdmol, reactant_rdmol)
    reactant = Molecule.from_rdkit(reactant_rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    product = Molecule.from_rdkit(product_rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True)

    reactant_top, reactant_pos = to_openmm_topology_positions(reactant)
    product_top, product_pos = to_openmm_topology_positions(product)

    reactant_ff = create_forcefield(reactant)
    reactant_modeller = app.Modeller(reactant_top, reactant_pos)
    reactant_modeller.addSolvent(
        reactant_ff,
        model="tip3p",
        boxSize=mm.Vec3(3.0, 3.0, 3.0) * unit.nanometer,
        neutralize=False,
    )
    water_top, water_pos = extract_water_only(reactant_modeller.topology, reactant_modeller.positions)

    product_modeller = app.Modeller(product_top, product_pos)
    product_modeller.add(water_top, water_pos)
    if reactant_modeller.topology.getPeriodicBoxVectors() is not None:
        product_modeller.topology.setPeriodicBoxVectors(reactant_modeller.topology.getPeriodicBoxVectors())

    reactant_state = parameterize_state(reactant_modeller.topology, reactant_modeller.positions, reactant)
    product_state = parameterize_state(product_modeller.topology, product_modeller.positions, product)
    reactant_state = minimize_state(reactant_state)
    product_state = minimize_state(product_state)
    builder = EVBSystemBuilder()
    builder.validate_compatibility(reactant_state, product_state)
    evb_system = builder.build_openmm_evb_system(reactant_state, product_state, delta_alpha=0.0, h12=0.0)
    evb_simulation = EVBSimulation(evb_system=evb_system, integrator=create_integrator(1.0, integrator_name="Verlet"))
    evb_result = evb_simulation.single_point()

    write_pdb(str(output_dir / "reactant_solution.pdb"), reactant_state.topology, reactant_state.positions_nm)
    write_pdb(str(output_dir / "product_solution.pdb"), product_state.topology, product_state.positions_nm)
    save_system_xml(output_dir / "reactant_solution.xml", reactant_state.system)
    save_system_xml(output_dir / "product_solution.xml", product_state.system)

    with (output_dir / "solution_reference_targets.json").open("w", encoding="utf-8") as handle:
        json.dump(REFERENCE_TARGETS.as_dict(), handle, indent=2)
    with (output_dir / "solution_reference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "natoms": int(reactant_state.system.getNumParticles()),
                "evb_singlepoint_kj_mol": evb_result.evb_energy,
                "state1_singlepoint_kj_mol": evb_result.energy1,
                "state2_singlepoint_kj_mol": evb_result.energy2,
            },
            handle,
            indent=2,
        )

    print(f"Wrote solvated solution reference systems to {output_dir}")
    print(json.dumps(REFERENCE_TARGETS.as_dict(), indent=2))



def build_complex_rdmol(mapped_smiles: str, transfer_host_map: int) -> Chem.Mol:
    rdmol = Chem.MolFromSmiles(mapped_smiles)
    if rdmol is None:
        raise ValueError(f"Could not parse mapped SMILES: {mapped_smiles}")
    rdmol = Chem.AddHs(rdmol)
    if AllChem.EmbedMolecule(rdmol, randomSeed=2026) != 0:
        raise RuntimeError("RDKit embedding failed for the aqueous reference complex.")
    AllChem.MMFFOptimizeMolecule(rdmol)

    mark_transfer_hydrogen(rdmol, transfer_host_map)
    reordered = Chem.RenumberAtoms(rdmol, canonical_atom_order(rdmol))
    return reordered



def mark_transfer_hydrogen(rdmol: Chem.Mol, host_map: int) -> None:
    for atom in rdmol.GetAtoms():
        if atom.GetAtomMapNum() == host_map:
            hydrogens = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
            if len(hydrogens) != 1:
                raise ValueError(f"Expected exactly one transferable H attached to mapped atom {host_map}.")
            hydrogens[0].SetAtomMapNum(TRANSFER_MAP)
            return
    raise ValueError(f"Could not find mapped atom {host_map} when labeling transferable hydrogen.")



def canonical_atom_order(rdmol: Chem.Mol) -> list[int]:
    by_label: dict[tuple, int] = {}
    heavy = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rdmol.GetAtoms() if atom.GetAtomicNum() != 1}
    for heavy_map in HEAVY_MAPS:
        by_label[("heavy", heavy_map)] = heavy[heavy_map]

    h_counts: dict[int, int] = {}
    for atom in rdmol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        if atom.GetAtomMapNum() == TRANSFER_MAP:
            by_label[("transfer", TRANSFER_MAP)] = atom.GetIdx()
            continue
        heavy_neighbors = [nbr.GetAtomMapNum() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1]
        if len(heavy_neighbors) != 1:
            raise ValueError("Hydrogen in reactive complex should have exactly one heavy-atom neighbor.")
        heavy_map = heavy_neighbors[0]
        h_counts[heavy_map] = h_counts.get(heavy_map, 0) + 1
        by_label[("h", heavy_map, h_counts[heavy_map])] = atom.GetIdx()

    ordered_labels = [("heavy", map_id) for map_id in HEAVY_MAPS]
    ordered_labels.extend(("h", heavy_map, ordinal) for heavy_map, ordinal in NONTRANSFER_H_LABELS)
    ordered_labels.append(("transfer", TRANSFER_MAP))
    return [by_label[label] for label in ordered_labels]



def _rdmol_positions_nm(rdmol: Chem.Mol) -> np.ndarray:
    conformer = rdmol.GetConformer()
    return np.asarray(
        [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z] for i in range(rdmol.GetNumAtoms())],
        dtype=float,
    ) * 0.1


def _set_rdmol_positions_nm(rdmol: Chem.Mol, coordinates_nm: np.ndarray) -> None:
    conformer = rdmol.GetConformer()
    for index, xyz in enumerate(coordinates_nm):
        conformer.SetAtomPosition(index, xyz * 10.0)


def _map_to_index(rdmol: Chem.Mol) -> dict[int, int]:
    return {atom.GetAtomMapNum(): atom.GetIdx() for atom in rdmol.GetAtoms() if atom.GetAtomMapNum()}


def place_acetate_near_reactive_center(rdmol: Chem.Mol) -> Chem.Mol:
    coordinates = _rdmol_positions_nm(rdmol)
    map_to_index = _map_to_index(rdmol)
    c_h_idx = map_to_index[10]
    o_acc_idx = map_to_index[16]
    h_idx = map_to_index[17]

    c_pos = coordinates[c_h_idx]
    h_pos = coordinates[h_idx]
    direction = h_pos - c_pos
    direction = direction / np.linalg.norm(direction)
    coordinates[o_acc_idx] = h_pos + 0.14 * direction
    coordinates[map_to_index[14]] = coordinates[o_acc_idx] + 0.12 * direction
    coordinates[map_to_index[13]] = coordinates[map_to_index[14]] + 0.15 * direction
    _set_rdmol_positions_nm(rdmol, coordinates)
    return rdmol



def align_to_reference(target: Chem.Mol, reference: Chem.Mol) -> Chem.Mol:
    target_xyz = _rdmol_positions_nm(target)
    reference_xyz = _rdmol_positions_nm(reference)
    target_center = target_xyz.mean(axis=0)
    reference_center = reference_xyz.mean(axis=0)
    _set_rdmol_positions_nm(target, target_xyz - target_center + reference_center)
    return target



def to_openmm_topology_positions(offmol: Molecule):
    topology = Topology.from_molecules([offmol]).to_openmm()
    positions = offmol.conformers[0].to_openmm()
    return topology, positions



def create_forcefield(offmol: Molecule) -> app.ForceField:
    forcefield = app.ForceField("amber/tip3p_standard.xml")
    if offmol.partial_charges is None:
        offmol.assign_partial_charges(partial_charge_method="gasteiger")
    generator = SMIRNOFFTemplateGenerator(molecules=[offmol], forcefield="openff-2.2.1")
    forcefield.registerTemplateGenerator(generator.generator)
    return forcefield



def extract_water_only(topology: app.Topology, positions) -> tuple[app.Topology, list[unit.Quantity]]:
    new_top = app.Topology()
    atom_map = {}
    selected_positions = []
    residue_set = []
    for residue in topology.residues():
        if residue.name not in {"HOH", "WAT"}:
            continue
        residue_set.append(residue)
    if not residue_set:
        raise ValueError("No water molecules were found after solvation.")
    chain_map = {}
    positions_list = list(positions)
    for residue in residue_set:
        source_chain = residue.chain
        chain = chain_map.setdefault(source_chain, new_top.addChain(source_chain.id))
        new_residue = new_top.addResidue(residue.name, chain)
        for atom in residue.atoms():
            new_atom = new_top.addAtom(atom.name, atom.element, new_residue)
            atom_map[atom] = new_atom
            selected_positions.append(positions_list[atom.index])
    for bond in topology.bonds():
        if bond.atom1 in atom_map and bond.atom2 in atom_map:
            new_top.addBond(atom_map[bond.atom1], atom_map[bond.atom2])
    if topology.getPeriodicBoxVectors() is not None:
        new_top.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    return new_top, selected_positions



def parameterize_state(topology: app.Topology, positions, offmol: Molecule) -> LoadedAmberState:
    forcefield = create_forcefield(offmol)
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
    )
    positions_nm = np.asarray(positions.value_in_unit(unit.nanometer))
    box_vectors_nm = None
    if topology.getPeriodicBoxVectors() is not None:
        box_vectors_nm = np.asarray([vec.value_in_unit(unit.nanometer) for vec in topology.getPeriodicBoxVectors()])
    atom_labels = []
    for atom in topology.atoms():
        atom_labels.append((atom.residue.chain.id, atom.residue.name, atom.name, atom.index))
    masses_amu = np.asarray([system.getParticleMass(i).value_in_unit(unit.amu) for i in range(system.getNumParticles())])
    return LoadedAmberState(
        prmtop_path="",
        inpcrd_path="",
        topology=topology,
        system=system,
        positions_nm=positions_nm,
        box_vectors_nm=box_vectors_nm,
        atom_labels=atom_labels,
        masses_amu=masses_amu,
    )



def save_system_xml(path: Path, system: mm.System) -> None:
    path.write_text(mm.XmlSerializer.serialize(system), encoding="utf-8")


def minimize_state(state: LoadedAmberState, max_iterations: int = 250) -> LoadedAmberState:
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(state.system, integrator)
    if state.box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in state.box_vectors_nm))
    context.setPositions(state.positions_nm * unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=max_iterations)
    positions_nm = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return LoadedAmberState(
        prmtop_path=state.prmtop_path,
        inpcrd_path=state.inpcrd_path,
        topology=state.topology,
        system=state.system,
        positions_nm=positions_nm,
        box_vectors_nm=state.box_vectors_nm,
        atom_labels=state.atom_labels,
        masses_amu=state.masses_amu,
    )


if __name__ == "__main__":
    main()
