from __future__ import annotations

import numpy as np

from ..config import CVDefinition
from ..openmm_backend import LoadedAmberState
from ..types import ValidationReport


def validate_diabatic_states(state1: LoadedAmberState, state2: LoadedAmberState, reactive_atoms: CVDefinition | None = None) -> ValidationReport:
    notes: list[str] = []
    compatible = True
    label_mismatch_count = 0
    mass_mismatch_count = 0
    box_mismatch = False
    if state1.system.getNumParticles() != state2.system.getNumParticles():
        compatible = False
        notes.append("Different particle counts.")
    if state1.atom_labels != state2.atom_labels:
        label_mismatch_count = sum(1 for left, right in zip(state1.atom_labels, state2.atom_labels) if left != right)
        notes.append(f"Different residue labels or atom labels ({label_mismatch_count} mismatches).")
    if not np.allclose(state1.masses_amu, state2.masses_amu, atol=1.0e-6):
        mass_mismatch_count = int(np.count_nonzero(np.abs(state1.masses_amu - state2.masses_amu) > 1.0e-6))
        compatible = False
        notes.append(f"Different particle masses ({mass_mismatch_count} mismatches).")
    if state1.atom_names != state2.atom_names:
        compatible = False
        notes.append("Atom names/order differ.")
    if state1.box_vectors_nm is None and state2.box_vectors_nm is not None:
        compatible = False
        notes.append("State 2 is periodic but state 1 is not.")
    if state1.box_vectors_nm is not None and state2.box_vectors_nm is None:
        compatible = False
        notes.append("State 1 is periodic but state 2 is not.")
    if state1.box_vectors_nm is not None and state2.box_vectors_nm is not None and not np.allclose(state1.box_vectors_nm, state2.box_vectors_nm, atol=1.0e-6):
        box_mismatch = True
        notes.append("Periodic boxes differ.")

    if reactive_atoms is None:
        has_reactive_atoms = False
    else:
        atom_count = state1.system.getNumParticles()
        indices = (reactive_atoms.donor, reactive_atoms.proton, reactive_atoms.acceptor)
        has_reactive_atoms = all(0 <= index < atom_count for index in indices)
        if not has_reactive_atoms:
            compatible = False
            notes.append("Reactive atom indices are out of range for the diabatic states.")

    return ValidationReport(
        compatible=compatible,
        state1_particles=state1.system.getNumParticles(),
        state2_particles=state2.system.getNumParticles(),
        state1_forces=[state1.system.getForce(i).__class__.__name__ for i in range(state1.system.getNumForces())],
        state2_forces=[state2.system.getForce(i).__class__.__name__ for i in range(state2.system.getNumForces())],
        periodic=state1.box_vectors_nm is not None and state2.box_vectors_nm is not None,
        has_reactive_atoms=has_reactive_atoms,
        label_mismatch_count=label_mismatch_count,
        mass_mismatch_count=mass_mismatch_count,
        box_mismatch=box_mismatch,
        notes=notes,
    )
