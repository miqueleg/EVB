#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-outputs/hg317_qregion_reproduction}
find "$OUT" -maxdepth 3 -name '*summary*.json' -o -name '*.log'
