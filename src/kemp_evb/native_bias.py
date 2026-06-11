from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import openmm
except ImportError:  # pragma: no cover
    openmm = None


@dataclass(slots=True)
class NativeBiasSnapshot:
    step: int
    time_ps: float
    gap_kj_mol: float
    bias_kj_mol: float
    e1_kj_mol: float
    e2_kj_mol: float
    e_common_kj_mol: float | None
    E1_kj_mol: float | None
    E2_kj_mol: float | None
    Eevb_unbiased_kj_mol: float
    Eevb_biased_kj_mol: float
    w1: float
    w2: float


class NativeGapBiasTable1D:
    """Mutable 1D kJ/mol bias table for the shifted EVB gap."""

    function_name = "gap_bias"

    def __init__(
        self,
        grid_min: float,
        grid_max: float,
        grid_width: int,
        values_kj_mol: np.ndarray | list[float] | None = None,
        out_of_grid: str = "clamp",
    ):
        if openmm is None:  # pragma: no cover
            raise ImportError("OpenMM is required for native tabulated EVB gap biasing.")
        if grid_width < 2:
            raise ValueError("Native gap bias grid_width must be at least 2.")
        if grid_max <= grid_min:
            raise ValueError("Native gap bias grid_max must be greater than grid_min.")
        if out_of_grid not in {"clamp", "reject"}:
            raise ValueError("out_of_grid must be 'clamp' or 'reject'.")
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.grid_width = int(grid_width)
        self.out_of_grid = out_of_grid
        self.grid = np.linspace(self.grid_min, self.grid_max, self.grid_width, dtype=float)
        if values_kj_mol is None:
            self.values_kj_mol = np.zeros(self.grid_width, dtype=float)
        else:
            values = np.asarray(values_kj_mol, dtype=float)
            if values.shape != (self.grid_width,):
                raise ValueError(f"Bias table has shape {values.shape}; expected {(self.grid_width,)}.")
            self.values_kj_mol = values.copy()
        self.function_index: int | None = None
        self._openmm_function = None

    @property
    def spacing(self) -> float:
        return float((self.grid_max - self.grid_min) / (self.grid_width - 1))

    def create_openmm_function(self):
        self._openmm_function = openmm.Continuous1DFunction(
            [float(value) for value in self.values_kj_mol],
            self.grid_min,
            self.grid_max,
        )
        return self._openmm_function

    def add_to_force(self, evb_force: Any) -> int:
        self.function_index = int(evb_force.addTabulatedFunction(self.function_name, self.create_openmm_function()))
        return self.function_index

    def set_values(self, values_kj_mol: np.ndarray | list[float]) -> None:
        values = np.asarray(values_kj_mol, dtype=float)
        if values.shape != (self.grid_width,):
            raise ValueError(f"Bias table has shape {values.shape}; expected {(self.grid_width,)}.")
        self.values_kj_mol = values.copy()
        if self._openmm_function is not None:
            self._openmm_function.setFunctionParameters(
                [float(value) for value in self.values_kj_mol],
                self.grid_min,
                self.grid_max,
            )

    def update_context(self, evb_force: Any, context: Any) -> None:
        if self._openmm_function is not None:
            self._openmm_function.setFunctionParameters(
                [float(value) for value in self.values_kj_mol],
                self.grid_min,
                self.grid_max,
            )
        evb_force.updateParametersInContext(context)

    def checked_gap(self, gap_kj_mol: float) -> float:
        gap = float(gap_kj_mol)
        if self.grid_min <= gap <= self.grid_max:
            return gap
        if self.out_of_grid == "reject":
            raise ValueError(
                f"EVB gap {gap:.6g} kJ/mol is outside native bias grid "
                f"[{self.grid_min:.6g}, {self.grid_max:.6g}] kJ/mol."
            )
        return min(max(gap, self.grid_min), self.grid_max)

    def evaluate(self, gap_kj_mol: float) -> float:
        gap = self.checked_gap(gap_kj_mol)
        return float(np.interp(gap, self.grid, self.values_kj_mol))

    def deposit_gaussian(self, center_kj_mol: float, height_kj_mol: float, sigma_kj_mol: float) -> float:
        if sigma_kj_mol <= 0.0:
            raise ValueError("Gaussian bias_width must be positive.")
        center = self.checked_gap(center_kj_mol)
        delta = float(height_kj_mol) * np.exp(-0.5 * ((self.grid - center) / float(sigma_kj_mol)) ** 2)
        self.values_kj_mol = self.values_kj_mol + delta
        return float(delta.max())

    def write_restart(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "grid_width": self.grid_width,
            "out_of_grid": self.out_of_grid,
            "values_kj_mol": [float(value) for value in self.values_kj_mol],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_restart(cls, path: str | Path) -> "NativeGapBiasTable1D":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            grid_min=float(payload["grid_min"]),
            grid_max=float(payload["grid_max"]),
            grid_width=int(payload["grid_width"]),
            values_kj_mol=payload["values_kj_mol"],
            out_of_grid=payload.get("out_of_grid", "clamp"),
        )

    def write_table_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gap_kj_mol", "bias_kj_mol"])
            for gap, bias in zip(self.grid, self.values_kj_mol):
                writer.writerow([float(gap), float(bias)])


class NativeWellTemperedGapMetadynamics1D:
    def __init__(
        self,
        table: NativeGapBiasTable1D,
        bias_width: float,
        height_kj_mol: float,
        bias_factor: float,
        temperature_k: float,
        frequency: int,
        save_frequency: int | None,
        bias_dir: str | Path,
        restart: bool = True,
    ):
        if bias_factor <= 1.0:
            raise ValueError("Well-tempered bias_factor must be greater than 1.")
        if frequency <= 0:
            raise ValueError("Native metadynamics frequency must be positive.")
        self.table = table
        self.bias_width = float(bias_width)
        self.height_kj_mol = float(height_kj_mol)
        self.bias_factor = float(bias_factor)
        self.temperature_k = float(temperature_k)
        self.frequency = int(frequency)
        self.save_frequency = None if save_frequency is None else int(save_frequency)
        self.bias_dir = Path(bias_dir)
        self.restart = bool(restart)
        self.bias_dir.mkdir(parents=True, exist_ok=True)
        self.restart_path = self.bias_dir / "native_gap_bias_state.json"
        self.table_path = self.bias_dir / "native_gap_bias_table.csv"
        self.update_times_s: list[float] = []
        self.deposition_count = 0

    @property
    def delta_temperature_k(self) -> float:
        return (self.bias_factor - 1.0) * self.temperature_k

    def effective_height(self, current_bias_kj_mol: float) -> float:
        beta_scale = 1.0 / (0.00831446261815324 * self.delta_temperature_k)
        return float(self.height_kj_mol * math.exp(-float(current_bias_kj_mol) * beta_scale))

    def maybe_deposit(self, step: int, gap_kj_mol: float, evb_force: Any, context: Any) -> bool:
        if step <= 0 or step % self.frequency != 0:
            return False
        import time

        start = time.perf_counter()
        current_bias = self.table.evaluate(gap_kj_mol)
        height = self.effective_height(current_bias)
        self.table.deposit_gaussian(gap_kj_mol, height, self.bias_width)
        self.table.update_context(evb_force, context)
        self.deposition_count += 1
        self.update_times_s.append(time.perf_counter() - start)
        if self.save_frequency is not None and step % self.save_frequency == 0:
            self.save_state()
        return True

    def save_state(self) -> None:
        self.table.write_restart(self.restart_path)
        self.table.write_table_csv(self.table_path)

    def load_restart_if_requested(self) -> bool:
        if self.restart and self.restart_path.exists():
            loaded = NativeGapBiasTable1D.from_restart(self.restart_path)
            if (
                loaded.grid_min != self.table.grid_min
                or loaded.grid_max != self.table.grid_max
                or loaded.grid_width != self.table.grid_width
            ):
                raise ValueError("Native gap bias restart grid does not match the current config.")
            self.table.set_values(loaded.values_kj_mol)
            return True
        return False

    def timing_report(self) -> dict[str, float | int | None]:
        total = float(sum(self.update_times_s))
        count = len(self.update_times_s)
        return {
            "bias_update_time_s": total,
            "average_time_per_bias_update_s": total / count if count else None,
            "number_of_bias_updates": count,
        }


class NativeGapBiasColvarWriter:
    header = [
        "step",
        "time_ps",
        "gap_kj_mol",
        "bias_kj_mol",
        "e1_kj_mol",
        "e2_kj_mol",
        "e_common_kj_mol",
        "E1_kj_mol",
        "E2_kj_mol",
        "Eevb_unbiased_kj_mol",
        "Eevb_biased_kj_mol",
        "w1",
        "w2",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(self.header)

    def write(self, snapshot: NativeBiasSnapshot) -> None:
        self._writer.writerow([getattr(snapshot, name) for name in self.header])
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "NativeGapBiasColvarWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
