#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../code" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLE_NAME="Pt-Ni_Pt-Ti_Pt-Y"

if [ -z "${JOB_NUM:-}" ]; then
  echo "JOB_NUM is required" >&2
  exit 1
fi

export CONDITION_FILE="${CONDITION_FILE:-${CODE_DIR}/condition_list.csv}"
export VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
export MODULE_LOADS="${MODULE_LOADS:-intel intel-mpi cuda}"

SEED="${SEED:-0}"
BETA="${BETA:-1.0}"
MAX_ITER="${MAX_ITER:-5}"
CALCULATOR="${CALCULATOR:-fairchem}"
WITH_VISUALIZATION="${WITH_VISUALIZATION:-1}"
WITH_ANALYSIS="${WITH_ANALYSIS:-1}"
GRID_X="${GRID_X:-4}"
GRID_Y="${GRID_Y:-4}"
GRID_Z="${GRID_Z:-6}"
INITIAL_NUM_STRUCTURES="${INITIAL_NUM_STRUCTURES:-255}"
GENERATED_NUM_STRUCTURES="${GENERATED_NUM_STRUCTURES:-255}"
BINARY_VARIANTS="${BINARY_VARIANTS:-Pt-Ni,Pt-Ti,Pt-Y}"
ALL_ELEMENTS="${ALL_ELEMENTS:-Pt,Ni,Ti,Y}"
MIN_SECONDARY_FRACTION="${MIN_SECONDARY_FRACTION:-0.010416666666666666}"
MAX_SECONDARY_FRACTION="${MAX_SECONDARY_FRACTION:-0.9895833333333334}"
SOLVENT_CORRECTION_YAML="${SOLVENT_CORRECTION_YAML:-${CODE_DIR}/solvent_correction.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXAMPLE_DIR}/results/seed_${SEED}/${JOB_NUM}}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/result}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/log}"
TEMP_DIR="${TEMP_DIR:-${OUTPUT_DIR}/temp}"

# Fill LABEL_THRESHOLD/BATCH_SIZE/MAX_EPOCH/LATENT_SIZE from CSV when omitted
if [ -z "${LABEL_THRESHOLD:-}" ] || [ -z "${BATCH_SIZE:-}" ] || [ -z "${MAX_EPOCH:-}" ] || [ -z "${LATENT_SIZE:-}" ]; then
  eval "$(python3 - <<'PY'
import csv
import os
from pathlib import Path

job_num = os.environ["JOB_NUM"]
csv_path = Path(os.environ["CONDITION_FILE"])
if not csv_path.exists():
    raise SystemExit(f"Condition file not found: {csv_path}")

row = None
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        if str(r.get("n")) == str(job_num):
            row = r
            break

if row is None:
    raise SystemExit(f"JOB_NUM {job_num} not found in {csv_path}")

print(f"export LABEL_THRESHOLD={row['label_threshold']}")
print(f"export BATCH_SIZE={row['batch_size']}")
print(f"export MAX_EPOCH={row['max_epoch']}")
print(f"export LATENT_SIZE={row['latent_size']}")
PY
)"
fi

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  for mod in ${MODULE_LOADS}; do
    if module load "${mod}"; then
      echo "[${EXAMPLE_NAME}] loaded module: ${mod}"
    else
      echo "[${EXAMPLE_NAME}] warning: failed to load module '${mod}'" >&2
    fi
  done
else
  echo "[${EXAMPLE_NAME}] warning: 'module' command is unavailable; skipping module load" >&2
fi

if [ -f "${VENV_PATH}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "[${EXAMPLE_NAME}] activated virtualenv: ${VENV_PATH}"
else
  echo "[${EXAMPLE_NAME}] warning: virtualenv not found at ${VENV_PATH}; using system python" >&2
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
mkdir -p "${LOG_DIR}"

python3 "${CODE_DIR}/run_workflow.py" \
  --seed "${SEED}" \
  --job-num "${JOB_NUM}" \
  --label-threshold "${LABEL_THRESHOLD}" \
  --batch-size "${BATCH_SIZE}" \
  --max-epoch "${MAX_EPOCH}" \
  --latent-size "${LATENT_SIZE}" \
  --beta "${BETA}" \
  --max-iter "${MAX_ITER}" \
  --calculator "${CALCULATOR}" \
  --with-visualization "${WITH_VISUALIZATION}" \
  --with-analysis "${WITH_ANALYSIS}" \
  --grid-x "${GRID_X}" \
  --grid-y "${GRID_Y}" \
  --grid-z "${GRID_Z}" \
  --initial-num-structures "${INITIAL_NUM_STRUCTURES}" \
  --generated-num-structures "${GENERATED_NUM_STRUCTURES}" \
  --binary-variants "${BINARY_VARIANTS}" \
  --all-elements "${ALL_ELEMENTS}" \
  --min-secondary-fraction "${MIN_SECONDARY_FRACTION}" \
  --max-secondary-fraction "${MAX_SECONDARY_FRACTION}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --temp-dir "${TEMP_DIR}" \
  --solvent-correction-yaml "${SOLVENT_CORRECTION_YAML}"
