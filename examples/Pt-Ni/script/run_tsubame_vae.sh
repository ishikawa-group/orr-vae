#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${EXAMPLE_DIR}/code"

export JOB_NUM="${JOB_NUM:-manual}"

# Resolve output paths early for qsub logs
export PYTHONPATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)/src:${PYTHONPATH:-}"
eval "$(python3 "${CODE_DIR}/build_plan.py" --shell)"
mkdir -p "${LOG_DIR}"

bash "${SCRIPT_DIR}/run_iterative_screening.sh"
