#!/usr/bin/env bash
set -euo pipefail

ENV_PY="${ENV_PY:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "$ENV_PY" ]]; then
  ENV_PY="$(command -v python)"
fi
export PYTHONPATH="${PYTHONPATH:-src}"
unset LD_LIBRARY_PATH

"$ENV_PY" scripts/check_openmm_plumed_cuda.py --require-cuda --skip-plumed

mkdir -p runs/hg317_evb_enhanced_sampling/logs
mkdir -p runs/hg317_evb_enhanced_sampling/configs
SHARED_BIAS_DIR="$PWD/outputs/hg317_evb_gap_metad/shared_bias"
rm -rf "$PWD/outputs/hg317_evb_gap_metad/rep01" "$PWD/outputs/hg317_evb_gap_metad/rep02" "$SHARED_BIAS_DIR" "$PWD/outputs/hg317_evb_gap_metad/summary"
mkdir -p "$SHARED_BIAS_DIR"

for rep in 01 02; do
  cp examples/hg317_evb_gap_metad.yaml "runs/hg317_evb_enhanced_sampling/configs/gap_metad_rep${rep}.yaml"
  sed -i \
    -e "s/hg317-evb-gap-metad/hg317-evb-gap-metad-rep${rep}/" \
    -e "s#outputs/hg317_evb_gap_metad#outputs/hg317_evb_gap_metad/rep${rep}#" \
    -e "s/seed: 31727/seed: 317${rep}/" \
    -e "s#bias_dir: bias#bias_dir: $SHARED_BIAS_DIR#" \
    "runs/hg317_evb_enhanced_sampling/configs/gap_metad_rep${rep}.yaml"
done

SEED1="outputs/hg317_evb_ready_tsfocused_gap_pmf_test/rep01/windows/u034/final_state.pdb"
SEED2="outputs/hg317_evb_ready_final_gap_pmf/rep03/windows/u069/final_state.pdb"
if [[ -f "$SEED1" ]]; then
  sed -i "/output_dir:/a start_coordinates: $SEED1" \
    runs/hg317_evb_enhanced_sampling/configs/gap_metad_rep01.yaml
fi
if [[ -f "$SEED2" ]]; then
  sed -i "/output_dir:/a start_coordinates: $SEED2" \
    runs/hg317_evb_enhanced_sampling/configs/gap_metad_rep02.yaml
fi

CUDA_VISIBLE_DEVICES=0 "$ENV_PY" -m evb.cli evb-gap-metad \
  --config runs/hg317_evb_enhanced_sampling/configs/gap_metad_rep01.yaml \
  > runs/hg317_evb_enhanced_sampling/logs/gap_metad_rep01_gpu0.log 2>&1 &
pid1=$!

CUDA_VISIBLE_DEVICES=1 "$ENV_PY" -m evb.cli evb-gap-metad \
  --config runs/hg317_evb_enhanced_sampling/configs/gap_metad_rep02.yaml \
  > runs/hg317_evb_enhanced_sampling/logs/gap_metad_rep02_gpu1.log 2>&1 &
pid2=$!

wait "$pid1"
wait "$pid2"
