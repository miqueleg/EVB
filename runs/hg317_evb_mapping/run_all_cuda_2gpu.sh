#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=src

python - <<'PY'
import openmm as mm
from openmm import unit
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print("OpenMM platforms:", ", ".join(names))
print("OpenMM version:", mm.version.version)
if "CUDA" not in names:
    raise SystemExit("CUDA platform is required for this production run.")
for device in ("0", "1"):
    system = mm.System()
    system.addParticle(1.0 * unit.amu)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    try:
        context = mm.Context(system, integrator, platform, {"DeviceIndex": device})
    except Exception as exc:
        raise SystemExit(f"CUDA device {device} preflight failed: {exc}")
    del context, integrator
    print(f"CUDA device {device} preflight: ok")
PY

mkdir -p runs/hg317_evb_mapping/logs

run_replica() {
  local cfg="$1"
  local gpu="$2"
  local rep
  rep="$(basename "$cfg" .yaml)"
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[$(date -Is)] GPU $gpu validate $rep"
  python -m evb.cli validate --config "$cfg" > "runs/hg317_evb_mapping/logs/${rep}_validate.log" 2>&1
  echo "[$(date -Is)] GPU $gpu sample-series $rep"
  python -m evb.cli sample-series --config "$cfg" > "runs/hg317_evb_mapping/logs/${rep}_sample.log" 2>&1
  echo "[$(date -Is)] GPU $gpu analyze $rep"
  python -m evb.cli analyze --config "$cfg" > "runs/hg317_evb_mapping/logs/${rep}_analyze.log" 2>&1
}

index=0
for cfg in runs/hg317_evb_mapping/configs/rep*.yaml; do
  gpu=$(( index % 2 ))
  run_replica "$cfg" "$gpu" &
  index=$(( index + 1 ))
  if (( index % 2 == 0 )); then
    wait
  fi
done
wait
