from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cv import proton_transfer_coordinate
from .evb import EVBHamiltonian, EVBParameters, EVBResult
from .openmm_backend import EVBOpenMMSystem, _require_openmm

try:
    import openmm
    from openmm import unit
except ImportError:  # pragma: no cover
    openmm = None
    unit = None


@dataclass(slots=True)
class SimulationSnapshot:
    step: int
    time_ps: float
    energy1: float
    energy2: float
    evb_energy: float
    weight1: float
    weight2: float
    cv: float | None


class EVBSimulation:
    def __init__(
        self,
        evb_system: EVBOpenMMSystem,
        integrator: Any,
        platform_name: str | None = None,
        cv_atoms: tuple[int, int, int] | None = None,
    ):
        _require_openmm()
        self.evb_system = evb_system
        self.integrator = integrator
        self.cv_atoms = cv_atoms
        if platform_name:
            platform = openmm.Platform.getPlatformByName(platform_name)
            self.context = openmm.Context(evb_system.system, integrator, platform)
        else:
            self.context = openmm.Context(evb_system.system, integrator)
        if evb_system.box_vectors_nm is not None:
            self.context.setPeriodicBoxVectors(*_to_box_vectors(evb_system.box_vectors_nm))
        self.context.setPositions(_to_openmm_positions(evb_system.positions_nm))

    @property
    def system(self):
        return self.evb_system.system

    @property
    def topology(self):
        return self.evb_system.topology

    @property
    def parameters(self) -> EVBParameters:
        return EVBParameters(
            delta_alpha=self.context.getParameter("delta_alpha"),
            h12=self.context.getParameter("h12"),
        )

    def set_positions(self, positions_nm: np.ndarray) -> None:
        self.context.setPositions(_to_openmm_positions(np.asarray(positions_nm)))

    def get_positions_nm(self) -> np.ndarray:
        state = self.context.getState(getPositions=True)
        return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    def set_velocities_to_temperature(self, temperature_k: float, seed: int | None = None) -> None:
        if seed is None:
            self.context.setVelocitiesToTemperature(temperature_k * unit.kelvin)
        else:
            self.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)

    def set_evb_parameters(self, parameters: EVBParameters) -> None:
        self.context.setParameter("delta_alpha", parameters.delta_alpha)
        self.context.setParameter("h12", parameters.h12)

    def single_point(self, positions_nm: np.ndarray | None = None) -> EVBResult:
        if positions_nm is not None:
            self.set_positions(positions_nm)
        state = self.context.getState(getEnergy=True, getForces=True)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        energy1, energy2 = self._state_energies()
        parameters = self.parameters
        _, weight1, weight2 = EVBHamiltonian(parameters).lower_eigenvalue(energy1, energy2)
        return EVBResult(
            energy1=energy1,
            energy2=energy2,
            e2_shifted=energy2 + parameters.delta_alpha,
            evb_energy=float(potential),
            weight1=weight1,
            weight2=weight2,
            forces=forces,
        )

    def minimize(self, tolerance_kjmol_per_mol_nm: float = 10.0, max_iterations: int = 0) -> list[SimulationSnapshot]:
        history = [self._snapshot(step=0)]
        openmm.LocalEnergyMinimizer.minimize(
            self.context,
            tolerance_kjmol_per_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
            max_iterations,
        )
        history.append(self._snapshot(step=max_iterations if max_iterations > 0 else 1))
        return history

    def run_md(
        self,
        steps: int,
        report_interval: int = 10,
        trajectory_writer=None,
        log_path: str | None = None,
    ) -> list[SimulationSnapshot]:
        history: list[SimulationSnapshot] = []
        log_handle = None
        log_writer = None
        if log_path:
            log_handle = open(log_path, "w", newline="", encoding="utf-8")
            log_writer = csv.writer(log_handle)
            log_writer.writerow(["step", "time_ps", "E1_kJmol", "E2_kJmol", "Eevb_kJmol", "w1", "w2", "cv"])
        try:
            for start in range(0, steps, report_interval):
                advance = min(report_interval, steps - start)
                self.integrator.step(advance)
                snapshot = self._snapshot(step=start + advance)
                history.append(snapshot)
                if log_writer:
                    log_writer.writerow([
                        snapshot.step,
                        snapshot.time_ps,
                        snapshot.energy1,
                        snapshot.energy2,
                        snapshot.evb_energy,
                        snapshot.weight1,
                        snapshot.weight2,
                        snapshot.cv,
                    ])
                if trajectory_writer is not None:
                    trajectory_writer(self.get_positions_nm(), snapshot.step)
        finally:
            if log_handle is not None:
                log_handle.close()
        return history

    def compute_cv(self, positions_nm: np.ndarray | None = None) -> float | None:
        if self.cv_atoms is None:
            return None
        coords = self.get_positions_nm() if positions_nm is None else positions_nm
        return proton_transfer_coordinate(coords, *self.cv_atoms)

    def _snapshot(self, step: int) -> SimulationSnapshot:
        result = self.single_point()
        time_ps = self._time_ps()
        return SimulationSnapshot(
            step=step,
            time_ps=time_ps,
            energy1=result.energy1,
            energy2=result.energy2,
            evb_energy=result.evb_energy,
            weight1=result.weight1,
            weight2=result.weight2,
            cv=self.compute_cv(),
        )

    def _state_energies(self) -> tuple[float, float]:
        values = [float(value) for value in self.evb_system.evb_force.getCollectiveVariableValues(self.context)]
        report = self.evb_system.energy_decomposition_report or {}
        if report.get("enabled") and len(values) >= 3:
            e_common, e1, e2 = values[:3]
            return e_common + e1, e_common + e2
        return values[0], values[1]

    def _time_ps(self) -> float:
        if hasattr(self.integrator, "getStepSize"):
            step_size = self.integrator.getStepSize().value_in_unit(unit.picoseconds)
            if hasattr(self.integrator, "getStepCount"):
                return float(self.integrator.getStepCount() * step_size)
        return 0.0


def create_integrator(
    timestep_fs: float,
    temperature_k: float = 300.0,
    friction_per_ps: float = 1.0,
    integrator_name: str = "LangevinMiddle",
):
    _require_openmm()
    dt = timestep_fs * unit.femtoseconds
    if integrator_name == "Verlet":
        return openmm.VerletIntegrator(dt)
    if integrator_name == "LangevinMiddle":
        return openmm.LangevinMiddleIntegrator(temperature_k * unit.kelvin, friction_per_ps / unit.picosecond, dt)
    raise ValueError(f"Unsupported integrator: {integrator_name}")


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _to_openmm_positions(positions_nm: np.ndarray):
    return np.asarray(positions_nm) * unit.nanometer


def _to_box_vectors(box_vectors_nm: np.ndarray):
    return tuple(box_vectors_nm[i] * unit.nanometer for i in range(3))
