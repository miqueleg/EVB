from __future__ import annotations

from dataclasses import dataclass

from ..config import EVBConfig


@dataclass(slots=True)
class MappingWindowSpec:
    window_id: str
    lambda_value: float


@dataclass(slots=True)
class GapWindowSpec:
    window_id: str
    center_kj_mol: float
    force_constant_kj_mol2: float


@dataclass(slots=True)
class ProtonTransferWindowSpec:
    window_id: str
    center_nm: float
    force_constant_kj_mol_nm2: float


def build_mapping_windows(config: EVBConfig) -> list[MappingWindowSpec]:
    lambda_values = config.sampling.windows.mapping.lambda_values
    if not lambda_values:
        raise ValueError("No mapping lambda values are defined in config.sampling.windows.mapping.lambda_values.")
    return [MappingWindowSpec(window_id=f"w{index:03d}", lambda_value=float(lambda_value)) for index, lambda_value in enumerate(lambda_values)]


def get_mapping_window(config: EVBConfig, window_id: str) -> MappingWindowSpec:
    windows = {window.window_id: window for window in build_mapping_windows(config)}
    try:
        return windows[window_id]
    except KeyError as exc:
        known = ", ".join(sorted(windows))
        raise ValueError(f"Unknown window_id '{window_id}'. Available windows: {known}") from exc


def build_gap_windows(config: EVBConfig) -> list[GapWindowSpec]:
    centers = config.sampling.windows.gap_umbrella.centers_kj_mol
    force_constant = config.sampling.windows.gap_umbrella.force_constant_kj_mol2
    if not centers:
        raise ValueError("No gap umbrella centers are defined in config.sampling.windows.gap_umbrella.centers_kj_mol.")
    if force_constant is None:
        raise ValueError("config.sampling.windows.gap_umbrella.force_constant_kj_mol2 is required for gap umbrella sampling.")
    return [
        GapWindowSpec(window_id=f"u{index:03d}", center_kj_mol=float(center), force_constant_kj_mol2=float(force_constant))
        for index, center in enumerate(centers)
    ]


def get_gap_window(config: EVBConfig, window_id: str) -> GapWindowSpec:
    windows = {window.window_id: window for window in build_gap_windows(config)}
    try:
        return windows[window_id]
    except KeyError as exc:
        known = ", ".join(sorted(windows))
        raise ValueError(f"Unknown window_id '{window_id}'. Available umbrella windows: {known}") from exc


def build_proton_transfer_windows(config: EVBConfig) -> list[ProtonTransferWindowSpec]:
    centers = config.sampling.windows.proton_transfer_umbrella.centers_nm
    force_constant = config.sampling.windows.proton_transfer_umbrella.force_constant_kj_mol_nm2
    if not centers:
        raise ValueError("No proton-transfer umbrella centers are defined in config.sampling.windows.proton_transfer_umbrella.centers_nm.")
    if force_constant is None:
        raise ValueError("config.sampling.windows.proton_transfer_umbrella.force_constant_kj_mol_nm2 is required for proton-transfer umbrella sampling.")
    return [
        ProtonTransferWindowSpec(window_id=f"p{index:03d}", center_nm=float(center), force_constant_kj_mol_nm2=float(force_constant))
        for index, center in enumerate(centers)
    ]


def get_proton_transfer_window(config: EVBConfig, window_id: str) -> ProtonTransferWindowSpec:
    windows = {window.window_id: window for window in build_proton_transfer_windows(config)}
    try:
        return windows[window_id]
    except KeyError as exc:
        known = ", ".join(sorted(windows))
        raise ValueError(f"Unknown window_id '{window_id}'. Available proton-transfer windows: {known}") from exc
