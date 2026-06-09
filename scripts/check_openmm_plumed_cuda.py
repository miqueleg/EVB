from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OpenMM CUDA and openmm-plumed/OPES availability.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if a CUDA context cannot be created.")
    parser.add_argument("--skip-plumed", action="store_true", help="Only check OpenMM/CUDA; do not import openmm-plumed.")
    args = parser.parse_args()
    report: dict[str, object] = {"ok": True, "checks": {}}
    try:
        import openmm
        from openmm import unit
    except Exception as exc:
        report["ok"] = False
        report["checks"]["openmm"] = {"ok": False, "error": repr(exc)}
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    report["checks"]["openmm"] = {"ok": True, "version": openmm.version.version}
    platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    report["checks"]["platforms"] = platforms
    cuda_ok = False
    if "CUDA" in platforms:
        cuda_ok = _check_cuda_context(openmm, unit, report)
    else:
        report["checks"]["cuda"] = {"ok": False, "error": "CUDA platform is not installed."}

    if args.skip_plumed:
        if args.require_cuda and not cuda_ok:
            report["ok"] = False
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ok"] else 1)

    try:
        from openmmplumed import PlumedForce
    except Exception as exc:
        report["checks"]["openmmplumed"] = {"ok": False, "error": repr(exc)}
        report["ok"] = False
    else:
        report["checks"]["openmmplumed"] = {"ok": True}
        report["checks"]["plumed_metad"] = _check_plumed_action(openmm, unit, PlumedForce, _metad_script())
        report["checks"]["plumed_opes"] = _check_plumed_action(openmm, unit, PlumedForce, _opes_script())
        if not report["checks"]["plumed_metad"]["ok"] or not report["checks"]["plumed_opes"]["ok"]:
            report["ok"] = False

    if args.require_cuda and not cuda_ok:
        report["ok"] = False
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


def _check_cuda_context(openmm, unit, report: dict[str, object]) -> bool:
    system = openmm.System()
    system.addParticle(39.9)
    system.addParticle(39.9)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 0.2, 100.0)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    try:
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
        context.setPositions([[0, 0, 0], [0.2, 0, 0]] * unit.nanometer)
        integrator.step(1)
        report["checks"]["cuda"] = {
            "ok": True,
            "platform": context.getPlatform().getName(),
            "device_index": context.getPlatform().getPropertyValue(context, "DeviceIndex"),
        }
        return True
    except Exception as exc:
        report["checks"]["cuda"] = {"ok": False, "error": repr(exc)}
        return False


def _check_plumed_action(openmm, unit, plumed_force_class, script: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        system = openmm.System()
        system.addParticle(39.9)
        system.addParticle(39.9)
        force = openmm.HarmonicBondForce()
        force.addBond(0, 1, 0.2, 100.0)
        system.addForce(force)
        script = script.replace("FILE=", f"FILE={temp_dir}/").replace("STATE_WFILE=", f"STATE_WFILE={temp_dir}/")
        try:
            system.addForce(plumed_force_class(script))
            integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
            context.setPositions([[0, 0, 0], [0.2, 0, 0]] * unit.nanometer)
            integrator.step(2)
            files = sorted(path.name for path in Path(temp_dir).iterdir())
            return {"ok": True, "files": files}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}


def _metad_script() -> str:
    return """d: DISTANCE ATOMS=1,2
metad: METAD ARG=d PACE=10 HEIGHT=0.1 SIGMA=0.02 BIASFACTOR=5 TEMP=300 FILE=HILLS
PRINT ARG=d,metad.bias FILE=COLVAR STRIDE=10
"""


def _opes_script() -> str:
    return """d: DISTANCE ATOMS=1,2
opes: OPES_METAD ARG=d PACE=10 SIGMA=0.02 BARRIER=5 TEMP=300 STATE_WFILE=STATE STATE_WSTRIDE=10
PRINT ARG=d,opes.bias FILE=COLVAR STRIDE=10
"""


if __name__ == "__main__":
    main()
