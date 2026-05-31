#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$HOME/.bashrc"
set -euo pipefail

python scripts/prepare_kemp_solvent_weekend_run.py

CONFIG_DIR="$REPO_ROOT/runs/weekend_kemp_solvent/configs"
LOG_DIR="$REPO_ROOT/runs/weekend_kemp_solvent/logs"
mkdir -p "$LOG_DIR"

for cfg in "$CONFIG_DIR"/rep*.yaml; do
  stem="$(basename "$cfg" .yaml)"
  echo "[$(date --iso-8601=seconds)] START $stem"
  conda run -n kemp-evb kemp-evb validate-states --config "$cfg" > "$LOG_DIR/${stem}_validate.log" 2>&1
  conda run -n kemp-evb kemp-evb sample-series --config "$cfg" > "$LOG_DIR/${stem}_sample.log" 2>&1
  conda run -n kemp-evb kemp-evb analyze-gap --config "$cfg" > "$LOG_DIR/${stem}_analyze.log" 2>&1
  echo "[$(date --iso-8601=seconds)] DONE $stem"
done

PYTHONPATH=src python scripts/summarize_kemp_solvent_weekend.py > "$LOG_DIR/summary.log" 2>&1
echo "[$(date --iso-8601=seconds)] ALL DONE"
