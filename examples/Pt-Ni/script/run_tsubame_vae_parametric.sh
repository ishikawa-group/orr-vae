#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:10:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../code" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLE_NAME="Pt-Ni"

if [ -z "${JOB_NUM:-}" ]; then
  echo "JOB_NUM is required" >&2
  exit 1
fi

export CONDITION_FILE="${CONDITION_FILE:-${CODE_DIR}/condition_list.csv}"
export VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
export MODULE_LOADS="${MODULE_LOADS:-intel intel-mpi cuda}"

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
eval "$(python3 "${CODE_DIR}/build_plan.py" --shell)"
mkdir -p "${LOG_DIR}"

python3 "${CODE_DIR}/run_iterative_screening.py"
