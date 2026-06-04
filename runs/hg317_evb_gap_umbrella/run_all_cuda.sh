#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=src

python - <<'PY'
import openmm as mm
from openmm import unit
try:
    import importlib.metadata as metadata
except ImportError:
    metadata = None
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print("OpenMM platforms:", ", ".join(names))
print("OpenMM version:", mm.version.version)
if metadata is not None:
    for package in ("cuda-nvrtc", "cuda-version"):
        try:
            print(f"{package}:", metadata.version(package))
        except metadata.PackageNotFoundError:
            pass
if "CUDA" not in names:
    raise SystemExit("CUDA platform is required for this production run.")
system = mm.System()
system.addParticle(1.0 * unit.amu)
integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
platform = mm.Platform.getPlatformByName("CUDA")
try:
    context = mm.Context(system, integrator, platform)
except Exception as exc:
    raise SystemExit(
        "CUDA platform is installed but no usable CUDA context is available. "
        "If the error is CUDA_ERROR_UNSUPPORTED_PTX_VERSION, the NVIDIA driver is too old "
        "for the CUDA/NVRTC runtime in this conda environment. "
        f"Original error: {exc}"
    )
del context, integrator
print("CUDA context preflight: ok")
PY

mkdir -p runs/hg317_evb_gap_umbrella/logs

for cfg in runs/hg317_evb_gap_umbrella/configs/rep*.yaml; do
  rep="$(basename "$cfg" .yaml)"
  echo "[$(date -Is)] validate $rep"
  python -m evb.cli validate --config "$cfg" > "runs/hg317_evb_gap_umbrella/logs/${rep}_validate.log" 2>&1
  echo "[$(date -Is)] sample-series $rep"
  python -m evb.cli sample-series --config "$cfg" > "runs/hg317_evb_gap_umbrella/logs/${rep}_sample.log" 2>&1
  echo "[$(date -Is)] analyze $rep"
  python -m evb.cli analyze --config "$cfg" > "runs/hg317_evb_gap_umbrella/logs/${rep}_analyze.log" 2>&1
done
