#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_NAME="Pt-Ni"

if [ -n "${ORR_VAE_EXAMPLE_DIR:-}" ] && [ -d "${ORR_VAE_EXAMPLE_DIR}" ]; then
  EXAMPLE_DIR="$(cd "${ORR_VAE_EXAMPLE_DIR}" && pwd)"
elif [ -d "${SCRIPT_DIR}/../code" ]; then
  EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [ -n "${SGE_O_WORKDIR:-}" ] && [ -d "${SGE_O_WORKDIR}/examples/${EXAMPLE_NAME}/code" ]; then
  EXAMPLE_DIR="$(cd "${SGE_O_WORKDIR}/examples/${EXAMPLE_NAME}" && pwd)"
else
  echo "[${EXAMPLE_NAME}] error: failed to resolve example directory." >&2
  echo "[${EXAMPLE_NAME}] set ORR_VAE_EXAMPLE_DIR or submit via submit_all_jobs.py." >&2
  exit 1
fi

if [ -n "${ORR_VAE_CODE_DIR:-}" ] && [ -d "${ORR_VAE_CODE_DIR}" ]; then
  CODE_DIR="$(cd "${ORR_VAE_CODE_DIR}" && pwd)"
else
  CODE_DIR="${EXAMPLE_DIR}/code"
fi

if [ -n "${ORR_VAE_ROOT:-}" ] && [ -d "${ORR_VAE_ROOT}" ]; then
  ROOT_DIR="$(cd "${ORR_VAE_ROOT}" && pwd)"
else
  ROOT_DIR="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
fi

if [ ! -d "${CODE_DIR}" ]; then
  echo "[${EXAMPLE_NAME}] error: code directory not found: ${CODE_DIR}" >&2
  exit 1
fi

if [ -z "${JOB_NUM:-}" ]; then
  echo "JOB_NUM is required" >&2
  exit 1
fi

required_vars=(
  JOB_NUM
  SEED LABEL_THRESHOLD BATCH_SIZE MAX_EPOCH LATENT_SIZE
  BETA MAX_ITER CALCULATOR WITH_VISUALIZATION WITH_ANALYSIS KEEP_TEMP
  GRID_X GRID_Y GRID_Z
  INITIAL_NUM_STRUCTURES GENERATED_NUM_STRUCTURES
  MIN_SECONDARY_FRACTION MAX_SECONDARY_FRACTION
  OUTPUT_DIR DATA_DIR RESULT_DIR LOG_DIR TEMP_DIR
  SOLVENT_CORRECTION_YAML
  VENV_PATH MODULE_LOADS
)

missing_vars=()
for var_name in "${required_vars[@]}"; do
  if [ -z "${!var_name:-}" ]; then
    missing_vars+=("${var_name}")
  fi
done

if [ "${#missing_vars[@]}" -gt 0 ]; then
  echo "[${EXAMPLE_NAME}] error: missing required environment variables:" >&2
  echo "  ${missing_vars[*]}" >&2
  echo "[${EXAMPLE_NAME}] submit via submit_all_jobs.py to populate all parameters." >&2
  exit 1
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
  --keep-temp "${KEEP_TEMP}" \
  --grid-x "${GRID_X}" \
  --grid-y "${GRID_Y}" \
  --grid-z "${GRID_Z}" \
  --initial-num-structures "${INITIAL_NUM_STRUCTURES}" \
  --generated-num-structures "${GENERATED_NUM_STRUCTURES}" \
  --min-secondary-fraction "${MIN_SECONDARY_FRACTION}" \
  --max-secondary-fraction "${MAX_SECONDARY_FRACTION}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --temp-dir "${TEMP_DIR}" \
  --solvent-correction-yaml "${SOLVENT_CORRECTION_YAML}"
