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
    raise SystemExit("CUDA platform is required for this run.")
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

run_root="runs/hg317_evb_ready_tsfocused_gap_pmf_test"
fit_report="outputs/hg317_evb_ready_no_relax/analysis/evb_reference_fit_from_irc.json"
mkdir -p "$run_root/logs"

run_replica() {
  local cfg="$1"
  local gpu="$2"
  local output_root="$3"
  local rep
  rep="$(basename "$cfg" .yaml)"
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[$(date -Is)] GPU $gpu validate $rep"
  python -m evb.cli validate --config "$cfg" > "$run_root/logs/${rep}_validate.log" 2>&1
  echo "[$(date -Is)] GPU $gpu sample-series $rep"
  python -m evb.cli sample-series --config "$cfg" > "$run_root/logs/${rep}_sample.log" 2>&1
  mkdir -p "$output_root/analysis"
  cp "$fit_report" "$output_root/analysis/evb_reference_fit_from_irc.json"
  echo "[$(date -Is)] GPU $gpu analyze $rep"
  python -m evb.cli analyze --config "$cfg" > "$run_root/logs/${rep}_analyze.log" 2>&1
  echo "[$(date -Is)] GPU $gpu done $rep"
}

run_replica "$run_root/configs/rep01.yaml" 0 "outputs/hg317_evb_ready_tsfocused_gap_pmf_test/rep01" &
pid0=$!
run_replica "$run_root/configs/rep02.yaml" 1 "outputs/hg317_evb_ready_tsfocused_gap_pmf_test/rep02" &
pid1=$!

wait "$pid0"
wait "$pid1"

echo "[$(date -Is)] HG3.17 EVB-ready TS-focused gap PMF test complete"
