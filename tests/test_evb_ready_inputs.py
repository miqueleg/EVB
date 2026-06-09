import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit

from kemp_evb.evb_inputs import _apply_zero_exclusion_union, _collect_reactive_exclusion_union


def _system_with_exclusions(exclusions):
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0)
    nonbonded = openmm.NonbondedForce()
    for _ in range(4):
        nonbonded.addParticle(0.1, 0.2, 0.1)
    for atom1, atom2 in exclusions:
        nonbonded.addException(atom1, atom2, 0.0, 0.1, 0.0)
    system.addForce(nonbonded)
    return system


def _exception_pairs(system):
    nonbonded = next(force for force in system.getForces() if isinstance(force, openmm.NonbondedForce))
    return {
        tuple(sorted(nonbonded.getExceptionParameters(index)[:2]))
        for index in range(nonbonded.getNumExceptions())
    }


def test_reactive_exclusions_are_mirrored_between_states():
    state1 = _system_with_exclusions([(0, 1)])
    state2 = _system_with_exclusions([(1, 2)])

    records = _collect_reactive_exclusion_union(state1, state2, reactive_atoms={0, 1, 2}, explicit_pairs=set())
    added1 = _apply_zero_exclusion_union(state1, records)
    added2 = _apply_zero_exclusion_union(state2, records)

    assert {(0, 1), (1, 2)} == {record.pair for record in records}
    assert added1 == {(1, 2)}
    assert added2 == {(0, 1)}
    assert _exception_pairs(state1) == {(0, 1), (1, 2)}
    assert _exception_pairs(state2) == {(0, 1), (1, 2)}


def test_nonzero_14_exceptions_are_not_mirrored_automatically():
    state1 = _system_with_exclusions([])
    state2 = _system_with_exclusions([])
    nonbonded = next(force for force in state2.getForces() if isinstance(force, openmm.NonbondedForce))
    nonbonded.addException(1, 3, 0.01 * unit.elementary_charge**2, 0.2 * unit.nanometer, 0.1 * unit.kilojoule_per_mole)

    records = _collect_reactive_exclusion_union(state1, state2, reactive_atoms={1}, explicit_pairs=set())

    assert records == []


def test_explicit_reactive_pair_is_added_even_without_existing_exception():
    state1 = _system_with_exclusions([])
    state2 = _system_with_exclusions([])

    records = _collect_reactive_exclusion_union(state1, state2, reactive_atoms=set(), explicit_pairs={(1, 2)})
    added1 = _apply_zero_exclusion_union(state1, records)
    added2 = _apply_zero_exclusion_union(state2, records)

    assert [record.pair for record in records] == [(1, 2)]
    assert added1 == {(1, 2)}
    assert added2 == {(1, 2)}
