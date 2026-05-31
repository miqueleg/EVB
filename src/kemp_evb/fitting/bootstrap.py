from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..config import EVBConfig
from ..evb import EVBParameters, calibrate_evb_parameters
from ..io import write_json


def fit_bootstrap_parameters(config: EVBConfig) -> EVBParameters:
    if config.calibration is None:
        raise ValueError("Bootstrap fitting requires a calibration section with MM and QM/MM targets.")
    cal = config.calibration
    required = [
        cal.e_mm_min1_state1,
        cal.e_mm_min1_state2,
        cal.e_mm_min2_state1,
        cal.e_mm_min2_state2,
        cal.e_mm_ts_state1,
        cal.e_mm_ts_state2,
    ]
    if any(value is None for value in required):
        raise ValueError("Bootstrap fitting requires explicit MM calibration energies for min1, min2, and TS.")
    params = calibrate_evb_parameters(
        e_mm_min1_state1=cal.e_mm_min1_state1,
        e_mm_min1_state2=cal.e_mm_min1_state2,
        e_mm_min2_state1=cal.e_mm_min2_state1,
        e_mm_min2_state2=cal.e_mm_min2_state2,
        e_mm_ts_state1=cal.e_mm_ts_state1,
        e_mm_ts_state2=cal.e_mm_ts_state2,
        e_qmmm_min1=cal.e_qmmm_min1,
        e_qmmm_min2=cal.e_qmmm_min2,
        e_qmmm_ts=cal.e_qmmm_ts,
    )
    output_dir = Path(config.output_dir) / "fitting"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "bootstrap_fit.json", asdict(params))
    return params
