#!/usr/bin/env bash
set -euo pipefail
OUTPUT=outputs/hg317_qregion_reproduction
PLATFORM=CUDA
MODE=production
REPLICAS=3
UMBRELLA_STEPS=200000
METAD_STEPS=1000000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2;;
    --platform) PLATFORM="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --replicas) REPLICAS="$2"; shift 2;;
    --umbrella-steps) UMBRELLA_STEPS="$2"; shift 2;;
    --metad-steps) METAD_STEPS="$2"; shift 2;;
    --resume|--skip-existing) shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUTPUT/logs"
echo "git_sha=$(git rev-parse HEAD 2>/dev/null || true)" | tee -a "$OUTPUT/logs/run.log"
echo "platform=$PLATFORM mode=$MODE workflow=metad" | tee -a "$OUTPUT/logs/run.log"
.venv/bin/evb hg317-qregion-reproduce \
  --config examples/hg317_evb_gap_metad.yaml \
  --reference examples/hg317_gxtb_reference_profile.yaml \
  --output "$OUTPUT" \
  --platform "$PLATFORM" \
  --mode "$MODE" \
  --workflow metad \
  --replicas "$REPLICAS" \
  --umbrella-steps "$UMBRELLA_STEPS" \
  --metad-steps "$METAD_STEPS" 2>&1 | tee -a "$OUTPUT/logs/run.log"
