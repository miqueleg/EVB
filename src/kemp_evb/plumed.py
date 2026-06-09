from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .config import PlumedSettings


class PlumedUnavailableError(ImportError):
    pass


def load_plumed_script(settings: PlumedSettings, base_dir: str | Path = ".") -> str:
    if settings.script_file:
        path = Path(settings.script_file)
        if not path.is_absolute():
            path = Path(base_dir) / path
        script = path.read_text(encoding="utf-8")
    else:
        script = settings.script or ""
    if settings.restart and "RESTART" not in script:
        script = "RESTART\n" + script
    if settings.output_colvar and "PRINT" not in script:
        script += f"\nPRINT ARG=* FILE={settings.output_colvar} STRIDE=100\n"
    return script.strip() + "\n"


def validate_plumed_script(settings: PlumedSettings, script: str) -> list[str]:
    """Return non-fatal warnings for common PLUMED/EVB setup mistakes."""
    warnings_out: list[str] = []
    mode = settings.mode.lower()
    upper = script.upper()
    if mode == "metad" and "METAD" not in upper:
        warnings_out.append("plumed.mode is 'metad' but the script does not contain a METAD action.")
    if mode in {"opes", "opes_metad"} and "OPES_METAD" not in upper:
        warnings_out.append("plumed.mode is 'opes' but the script does not contain an OPES_METAD action.")
    if "OPES_METAD" in upper and "KERNELS=" in upper:
        warnings_out.append(
            "This PLUMED build may reject OPES_METAD KERNELS=...; use STATE_WFILE/STATE_RFILE for restart "
            "and consult your PLUMED version for kernel-output keywords."
        )
    if "METAD " in upper and ".BIAS" in upper and ":" not in script.split("METAD", 1)[0].splitlines()[-1]:
        warnings_out.append("METAD bias output requires a labeled action, e.g. 'metad: METAD ...'.")
    return warnings_out


def attach_plumed_force(system: Any, settings: PlumedSettings, base_dir: str | Path = ".") -> Any:
    try:
        from openmmplumed import PlumedForce
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise PlumedUnavailableError("Install the 'plumed' extra to use openmm-plumed.") from exc

    warnings.warn(
        "PLUMED atom indices are 1-based; OpenMM/Python atom indices are 0-based. "
        "This package currently supports PLUMED biases on geometrical CVs only, not the internal EVB energy gap.",
        UserWarning,
        stacklevel=2,
    )
    script = load_plumed_script(settings, base_dir=base_dir)
    for message in validate_plumed_script(settings, script):
        warnings.warn(message, UserWarning, stacklevel=2)
    force = PlumedForce(script)
    system.addForce(force)
    return force


PLUMED_TEMPLATES: dict[str, str] = {
    "distance": """# PLUMED indices are 1-based.
d1: DISTANCE ATOMS=1,2
PRINT ARG=d1 FILE=COLVAR STRIDE=100
""",
    "difference_of_distances": """# Difference of distances for bond breaking/forming.
d_break: DISTANCE ATOMS=1,2
d_form: DISTANCE ATOMS=2,3
dd: COMBINE ARG=d_break,d_form COEFFICIENTS=1,-1 PERIODIC=NO
PRINT ARG=d_break,d_form,dd FILE=COLVAR STRIDE=100
""",
    "2d_bond_form_break": """d_break: DISTANCE ATOMS=1,2
d_form: DISTANCE ATOMS=2,3
PRINT ARG=d_break,d_form FILE=COLVAR STRIDE=100
""",
    "metad": """d1: DISTANCE ATOMS=1,2
metad: METAD ARG=d1 PACE=500 HEIGHT=1.2 SIGMA=0.02 BIASFACTOR=10 TEMP=300 FILE=HILLS
PRINT ARG=d1,metad.bias FILE=COLVAR STRIDE=100
""",
    "opes_metad": """d1: DISTANCE ATOMS=1,2
opes: OPES_METAD ARG=d1 PACE=500 SIGMA=0.02 BARRIER=40 TEMP=300 STATE_WFILE=STATE STATE_WSTRIDE=1000
PRINT ARG=d1,opes.bias FILE=COLVAR STRIDE=100
""",
}
