#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${EXAMPLE_DIR}/code"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
eval "$(python3 "${CODE_DIR}/build_plan.py" --shell)"

mkdir -p "${DATA_DIR}" "${RESULT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/temp"

echo "[Pt-Ni_Pt-Ti_Pt-Y] output_dir=${OUTPUT_DIR}"
echo "[Pt-Ni_Pt-Ti_Pt-Y] data_dir=${DATA_DIR}"
echo "[Pt-Ni_Pt-Ti_Pt-Y] result_dir=${RESULT_DIR}"

echo "[Pt-Ni_Pt-Ti_Pt-Y] binary variants: ${BINARY_VARIANTS}"
python3 "${CODE_DIR}/initial_generation.py" \
  --output_dir "${DATA_DIR}" \
  --num "${INITIAL_NUM_STRUCTURES}" \
  --seed "${SEED}"

for iter_idx in $(seq 0 "${MAX_ITER}"); do
  echo "[Pt-Ni_Pt-Ti_Pt-Y] ===== iter ${iter_idx}/${MAX_ITER} : evaluate -> train -> generate ====="

  python3 -m orr_vae.cli.main calc-orr run-all \
    --iter "${iter_idx}" \
    --base_dir "${OUTPUT_DIR}" \
    --base_data_dir "${DATA_DIR}" \
    --temp_base_dir "${OUTPUT_DIR}/temp" \
    --calculator "${CALCULATOR}" \
    --solvent_correction_yaml_path "${SOLVENT_CORRECTION_YAML}"

  python3 -m orr_vae.cli.main train-cvae \
    --iter "${iter_idx}" \
    --label_threshold "${LABEL_THRESHOLD}" \
    --batch_size "${BATCH_SIZE}" \
    --max_epoch "${MAX_EPOCH}" \
    --beta "${BETA}" \
    --latent_size "${LATENT_SIZE}" \
    --seed "${SEED}" \
    --base_data_path "${DATA_DIR}" \
    --result_base_path "${RESULT_DIR}"

  python3 -m orr_vae.cli.main generate-structures \
    --iter "${iter_idx}" \
    --num "${GENERATED_NUM_STRUCTURES}" \
    --overpotential_condition 1 \
    --alloy_stability_condition 1 \
    --latent_size "${LATENT_SIZE}" \
    --seed "${SEED}" \
    --output_dir "${DATA_DIR}" \
    --result_dir "${RESULT_DIR}"

  if [ "${WITH_VISUALIZATION}" = "1" ]; then
    python3 -m orr_vae.cli.main visualize-latent \
      --iter "${iter_idx}" \
      --latent_size "${LATENT_SIZE}" \
      --seed "${SEED}" \
      --data_dir "${DATA_DIR}" \
      --result_dir "${RESULT_DIR}"
  fi

done

if [ "${WITH_ANALYSIS}" = "1" ]; then
  python3 -m orr_vae.cli.main analyze \
    --iter "${MAX_ITER}" \
    --base_path "${DATA_DIR}" \
    --output_path "${RESULT_DIR}/figure"
fi

echo "[Pt-Ni_Pt-Ti_Pt-Y] completed: ${OUTPUT_DIR}"
