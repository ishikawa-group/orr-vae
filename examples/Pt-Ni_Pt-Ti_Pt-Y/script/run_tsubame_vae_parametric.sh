#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
set -euo pipefail

if [ -z "${JOB_NUM:-}" ]; then
  echo "JOB_NUM is required" >&2
  exit 1
fi

: "${SEED:=0}"
: "${LABEL_THRESHOLD:=0.3}"
: "${BATCH_SIZE:=16}"
: "${MAX_EPOCH:=200}"
: "${LATENT_SIZE:=32}"

export BETA="${BETA:-1}"
export NUM_STRUCTURES="${NUM_STRUCTURES:-128}"
export MAX_ITER="${MAX_ITER:-5}"
export CALCULATOR="${CALCULATOR:-fairchem}"

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_tsubame_vae.sh"
