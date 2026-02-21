#!/usr/bin/env bash
set -euo pipefail

: "${LABEL_THRESHOLD:=0.3}"
: "${BATCH_SIZE:=16}"
: "${MAX_EPOCH:=200}"
: "${LATENT_SIZE:=32}"
: "${BETA:=1}"
: "${SEED:=0}"
: "${NUM_STRUCTURES:=128}"
: "${MAX_ITER:=5}"
: "${CALCULATOR:=fairchem}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python3 -m orr_vae.cli.main run-pipeline \
  --data_dir "${DATA_DIR}" \
  --result_dir "${RESULT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --label_threshold "${LABEL_THRESHOLD}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epoch "${MAX_EPOCH}" \
  --latent_size "${LATENT_SIZE}" \
  --beta "${BETA}" \
  --num_structures "${NUM_STRUCTURES}" \
  --max_iter "${MAX_ITER}" \
  --calculator "${CALCULATOR}"
