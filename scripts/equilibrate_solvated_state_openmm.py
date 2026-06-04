from __future__ import annotations

from pathlib import Path

import openmm as mm
from openmm import unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, PDBFile, Simulation


def _read_positions(inpcrd_path: str, coordinates_pdb: str | None):
    if coordinates_pdb:
        pdb = PDBFile(coordinates_pdb)
        return pdb.positions, pdb.topology.getPeriodicBoxVectors()
    inpcrd = AmberInpcrdFile(inpcrd_path)
    return inpcrd.positions, inpcrd.boxVectors


def _restrained_system(prmtop: AmberPrmtopFile, restraint_k_kcal_a2: float):
    system = prmtop.createSystem(
        nonbondedMethod=mm.app.PME,
        constraints=None,
    )
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", restraint_k_kcal_a2 * 418.4)  # kcal/mol/A^2 -> kJ/mol/nm^2
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for atom in prmtop.topology.atoms():
        if atom.residue.name != "WAT":
            force.addParticle(atom.index, [0.0, 0.0, 0.0])
    system.addForce(force)
    return system, force


def _inject_reference_positions(force, positions):
    positions_nm = positions.value_in_unit(unit.nanometer)
    for idx in range(force.getNumParticles()):
        atom_index, _ = force.getParticleParameters(idx)
        pos = positions_nm[atom_index]
        force.setParticleParameters(idx, atom_index, [float(pos[0]), float(pos[1]), float(pos[2])])


def _write_pdb(path: Path, topology, positions):
    with path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, positions, handle)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Restrained OpenMM minimization/heating/equilibration helper for solvated Amber states.")
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--inpcrd", required=True)
    parser.add_argument("--coordinates-pdb")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--minimize-only", action="store_true")
    parser.add_argument("--restraint-k-kcal-a2", type=float, default=250.0)
    parser.add_argument("--minimize-max-iters", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prmtop = AmberPrmtopFile(args.prmtop)
    positions, box_vectors = _read_positions(args.inpcrd, args.coordinates_pdb)
    system, restraint_force = _restrained_system(prmtop, args.restraint_k_kcal_a2)
    _inject_reference_positions(restraint_force, positions)

    integrator = mm.LangevinMiddleIntegrator(5.0 * unit.kelvin, 1.0 / unit.picosecond, 0.0001 * unit.picoseconds)
    sim = None
    if any(mm.Platform.getPlatform(i).getName() == "CUDA" for i in range(mm.Platform.getNumPlatforms())):
        try:
            sim = Simulation(prmtop.topology, system, integrator, mm.Platform.getPlatformByName("CUDA"))
        except Exception:
            sim = None
    if sim is None:
        sim = Simulation(prmtop.topology, system, integrator, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(positions)
    if box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.minimizeEnergy(maxIterations=args.minimize_max_iters)
    state1 = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    _write_pdb(outdir / f"{args.prefix}_minimized.pdb", prmtop.topology, state1.getPositions())
    (outdir / f"{args.prefix}_minimization_summary.txt").write_text(
        f"initial_potential_kj_mol: {e0}\nminimized_potential_kj_mol: {e1}\n",
        encoding="utf-8",
    )
    if not args.minimize_only:
        raise NotImplementedError("This helper currently supports minimize-only mode.")


if __name__ == "__main__":
    main()
