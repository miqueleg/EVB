from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GapSample:
    frame: int
    step: int
    time_ps: float
    energy1_kj_mol: float
    energy2_kj_mol: float
    delta_e_kj_mol: float
    delta_e_shifted_kj_mol: float
    evb_energy_kj_mol: float
    weight1: float
    weight2: float


@dataclass(slots=True)
class FrameObservables:
    gap: GapSample
    distances_nm: dict[str, float] = field(default_factory=dict)
    reaction_coordinates_nm: dict[str, float] = field(default_factory=dict)
    proton_transfer_rc_nm: float | None = None
    proton_transfer_event: bool = False


@dataclass(slots=True)
class ValidationReport:
    compatible: bool
    state1_particles: int
    state2_particles: int
    state1_forces: list[str]
    state2_forces: list[str]
    periodic: bool
    has_reactive_atoms: bool
    label_mismatch_count: int = 0
    mass_mismatch_count: int = 0
    box_mismatch: bool = False
    notes: list[str] = field(default_factory=list)
