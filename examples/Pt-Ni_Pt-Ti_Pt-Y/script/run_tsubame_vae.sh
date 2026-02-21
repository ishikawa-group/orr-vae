#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${EXAMPLE_DIR}/code"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JOB_NUM="${JOB_NUM:-manual}"
export VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
export MODULE_LOADS="${MODULE_LOADS:-intel intel-mpi cuda}"

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  for mod in ${MODULE_LOADS}; do
    if module load "${mod}"; then
      echo "[Pt-Ni_Pt-Ti_Pt-Y] loaded module: ${mod}"
    else
      echo "[Pt-Ni_Pt-Ti_Pt-Y] warning: failed to load module '${mod}'" >&2
    fi
  done
else
  echo "[Pt-Ni_Pt-Ti_Pt-Y] warning: 'module' command is unavailable; skipping module load" >&2
fi

if [ -f "${VENV_PATH}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  echo "[Pt-Ni_Pt-Ti_Pt-Y] activated virtualenv: ${VENV_PATH}"
else
  echo "[Pt-Ni_Pt-Ti_Pt-Y] warning: virtualenv not found at ${VENV_PATH}; using system python" >&2
fi

# Resolve output paths early for qsub logs
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
eval "$(python3 "${CODE_DIR}/build_plan.py" --shell)"
mkdir -p "${LOG_DIR}"

bash "${SCRIPT_DIR}/run_iterative_screening.sh"
