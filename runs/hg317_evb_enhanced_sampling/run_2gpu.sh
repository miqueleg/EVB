#!/usr/bin/env bash
set -euo pipefail

ENV_PY="${ENV_PY:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "$ENV_PY" ]]; then
  ENV_PY="$(command -v python)"
fi
export PYTHONPATH="${PYTHONPATH:-src}"
unset LD_LIBRARY_PATH

"$ENV_PY" scripts/check_openmm_plumed_cuda.py --require-cuda

mkdir -p runs/hg317_evb_enhanced_sampling/logs

run_one() {
  local gpu="$1"
  local command="$2"
  local config="$3"
  local label="$4"
  export CUDA_VISIBLE_DEVICES="$gpu"
  "$ENV_PY" -m evb.cli "$command" --config "$config" \
    > "runs/hg317_evb_enhanced_sampling/logs/${label}.log" 2>&1
}

run_one 0 evb-metad examples/hg317_evb_plumed_metad.yaml plumed_metad_gpu0 &
pid1=$!
run_one 1 evb-opes examples/hg317_evb_plumed_opes.yaml plumed_opes_gpu1 &
pid2=$!

wait "$pid1"
wait "$pid2"
