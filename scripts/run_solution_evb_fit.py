from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

from kemp_evb.evb import EVBParameters
from kemp_evb.openmm_backend import EVBSystemBuilder, LoadedAmberState
from kemp_evb.simulation import EVBSimulation, create_integrator
from kemp_evb.solution_reference import REFERENCE_TARGETS

OUTPUT_DIR = Path("outputs/solution_reference")
REACTANT_PDB = OUTPUT_DIR / "reactant_solution.pdb"
PRODUCT_PDB = OUTPUT_DIR / "product_solution.pdb"
REACTANT_XML = OUTPUT_DIR / "reactant_solution.xml"
PRODUCT_XML = OUTPUT_DIR / "product_solution.xml"

DONOR_INDEX = 9
ACCEPTOR_INDEX = 15
PROTON_INDEX = 22
TARGET_BARRIER_KJ_MOL = REFERENCE_TARGETS.barrier_kcal_mol * 4.184
N_WINDOWS = 21


def main() -> None:
    reactant_state = load_state(REACTANT_PDB, REACTANT_XML)
    product_state = load_state(PRODUCT_PDB, PRODUCT_XML)
    reactant_min = reactant_state
    product_min = product_state

    e1_r, _ = evaluate_state(reactant_min.system, reactant_min.positions_nm, reactant_min.box_vectors_nm)
    e2_p, _ = evaluate_state(product_min.system, product_min.positions_nm, product_min.box_vectors_nm)
    delta_alpha = e1_r - e2_p

    builder = EVBSystemBuilder()
    builder.validate_compatibility(reactant_min, product_min)

    lambdas = np.linspace(0.0, 1.0, N_WINDOWS)
    no_coupling_profile = compute_profile(builder, reactant_min, product_min, EVBParameters(delta_alpha=delta_alpha, h12=0.0), lambdas)
    h12 = fit_h12_from_crossing(no_coupling_profile)
    fitted = EVBParameters(delta_alpha=delta_alpha, h12=h12)

    profile = compute_profile(builder, reactant_min, product_min, fitted, lambdas)
    profile_no_alpha = compute_profile(builder, reactant_min, product_min, EVBParameters(delta_alpha=0.0, h12=0.0), lambdas)

    write_outputs(delta_alpha, h12, profile, profile_no_alpha)



def load_state(pdb_path: Path, xml_path: Path) -> LoadedAmberState:
    pdb = app.PDBFile(str(pdb_path))
    system = mm.XmlSerializer.deserialize(xml_path.read_text(encoding="utf-8"))
    positions_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    box_vectors_nm = None
    if pdb.topology.getPeriodicBoxVectors() is not None:
        box_vectors_nm = np.asarray([vec.value_in_unit(unit.nanometer) for vec in pdb.topology.getPeriodicBoxVectors()])
    atom_labels = [(atom.residue.chain.id, atom.residue.name, atom.name, atom.index) for atom in pdb.topology.atoms()]
    masses_amu = np.asarray([system.getParticleMass(i).value_in_unit(unit.amu) for i in range(system.getNumParticles())])
    return LoadedAmberState("", "", pdb.topology, system, positions_nm, box_vectors_nm, atom_labels, masses_amu)



def evaluate_state(system: mm.System, positions_nm: np.ndarray, box_vectors_nm: np.ndarray | None):
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator)
    if box_vectors_nm is not None:
        context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in box_vectors_nm))
    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    return float(energy), np.asarray(forces)



def proton_transfer_cv(positions_nm: np.ndarray) -> float:
    donor = positions_nm[DONOR_INDEX]
    proton = positions_nm[PROTON_INDEX]
    acceptor = positions_nm[ACCEPTOR_INDEX]
    return float(np.linalg.norm(donor - proton) - np.linalg.norm(acceptor - proton))



def fit_h12_from_crossing(profile: list[dict]) -> float:
    reactant_energy = profile[0]["Eevb_kj_mol"]
    crossing = min(profile, key=lambda point: abs(point["E1_kj_mol"] - point["E2_shifted_kj_mol"]))
    target_ts_energy = reactant_energy + TARGET_BARRIER_KJ_MOL
    e1 = crossing["E1_kj_mol"]
    e2_shifted = crossing["E2_shifted_kj_mol"]
    average = 0.5 * (e1 + e2_shifted)
    half_gap = 0.5 * (e1 - e2_shifted)
    root_target = average - target_ts_energy
    h_sq = max(0.0, root_target * root_target - half_gap * half_gap)
    return float(np.sqrt(h_sq))



def compute_profile(builder: EVBSystemBuilder, reactant: LoadedAmberState, product: LoadedAmberState, parameters: EVBParameters, lambdas: np.ndarray):
    evb_system = builder.build_openmm_evb_system(reactant, product, parameters.delta_alpha, parameters.h12)
    simulation = EVBSimulation(evb_system=evb_system, integrator=create_integrator(1.0, integrator_name="Verlet"))
    if reactant.box_vectors_nm is not None:
        simulation.context.setPeriodicBoxVectors(*(vec * unit.nanometer for vec in reactant.box_vectors_nm))
    profile = []
    for lam in lambdas:
        positions = (1.0 - lam) * reactant.positions_nm + lam * product.positions_nm
        simulation.set_positions(positions)
        result = simulation.single_point()
        profile.append(
            {
                "lambda": float(lam),
                "cv_nm": proton_transfer_cv(positions),
                "E1_kj_mol": result.energy1,
                "E2_kj_mol": result.energy2,
                "E2_shifted_kj_mol": result.e2_shifted,
                "Eevb_kj_mol": result.evb_energy,
            }
        )
    return profile



def write_outputs(delta_alpha: float, h12: float, profile: list[dict], profile_no_alpha: list[dict]) -> None:
    energies = [point["Eevb_kj_mol"] for point in profile]
    ts_index = int(np.argmax(energies))
    e0_fit = profile[0]["Eevb_kj_mol"]
    e0_no_alpha = profile_no_alpha[0]["Eevb_kj_mol"]
    summary = {
        "delta_alpha_kj_mol": delta_alpha,
        "h12_kj_mol": h12,
        "target_barrier_kj_mol": TARGET_BARRIER_KJ_MOL,
        "target_barrier_kcal_mol": REFERENCE_TARGETS.barrier_kcal_mol,
        "reactant_reference_energy_kj_mol": e0_fit,
        "fitted_barrier_kj_mol": max(energies) - e0_fit,
        "ts_index": ts_index,
        "ts_lambda": profile[ts_index]["lambda"],
        "ts_cv_nm": profile[ts_index]["cv_nm"],
        "ts_relative_energy_kj_mol": profile[ts_index]["Eevb_kj_mol"] - e0_fit,
        "ts_relative_energy_kcal_mol": (profile[ts_index]["Eevb_kj_mol"] - e0_fit) / 4.184,
    }
    with (OUTPUT_DIR / "evb_fit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (OUTPUT_DIR / "evb_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "window",
            "lambda",
            "cv_nm",
            "E1_kj_mol",
            "E2_kj_mol",
            "E2_shifted_kj_mol",
            "Eevb_fitted_kj_mol",
            "Eevb_fitted_rel_kj_mol",
            "Eevb_no_alpha_kj_mol",
            "Eevb_no_alpha_rel_kj_mol",
        ])
        for index, (fitted, no_alpha) in enumerate(zip(profile, profile_no_alpha)):
            writer.writerow([
                index,
                fitted["lambda"],
                fitted["cv_nm"],
                fitted["E1_kj_mol"],
                fitted["E2_kj_mol"],
                fitted["E2_shifted_kj_mol"],
                fitted["Eevb_kj_mol"],
                fitted["Eevb_kj_mol"] - e0_fit,
                no_alpha["Eevb_kj_mol"],
                no_alpha["Eevb_kj_mol"] - e0_no_alpha,
            ])


if __name__ == "__main__":
    main()
