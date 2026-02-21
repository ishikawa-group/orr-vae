#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${EXAMPLE_DIR}/code"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
eval "$(python3 "${CODE_DIR}/build_plan.py" --shell)"

mkdir -p "${DATA_DIR}" "${RESULT_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}/temp"

echo "[Pt-Ni] output_dir=${OUTPUT_DIR}"
echo "[Pt-Ni] data_dir=${DATA_DIR}"
echo "[Pt-Ni] result_dir=${RESULT_DIR}"

python3 "${CODE_DIR}/initial_generation.py" \
  --output_dir "${DATA_DIR}" \
  --num "${INITIAL_NUM_STRUCTURES}" \
  --seed "${SEED}"

python3 - <<'PY'
import os
import subprocess
import sys

from tqdm.auto import tqdm

env = os.environ
max_iter = int(env["MAX_ITER"])
with_visualization = env.get("WITH_VISUALIZATION", "0") == "1"

progress = tqdm(
    range(max_iter + 1),
    desc="Pt-Ni workflow",
    unit="iter",
    dynamic_ncols=True,
    disable=False,
)

for iter_idx in progress:
    print(
        f"[Pt-Ni] ===== iter {iter_idx}/{max_iter} : evaluate -> train -> generate =====",
        flush=True,
    )
    progress.set_postfix(iter=iter_idx, stage="evaluate")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "orr_vae.cli.main",
            "calc-orr",
            "run-all",
            "--iter",
            str(iter_idx),
            "--base_dir",
            env["OUTPUT_DIR"],
            "--base_data_dir",
            env["DATA_DIR"],
            "--temp_base_dir",
            f'{env["OUTPUT_DIR"]}/temp',
            "--calculator",
            env["CALCULATOR"],
            "--solvent_correction_yaml_path",
            env["SOLVENT_CORRECTION_YAML"],
        ],
        check=True,
    )

    progress.set_postfix(iter=iter_idx, stage="train")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "orr_vae.cli.main",
            "train-cvae",
            "--iter",
            str(iter_idx),
            "--label_threshold",
            env["LABEL_THRESHOLD"],
            "--batch_size",
            env["BATCH_SIZE"],
            "--max_epoch",
            env["MAX_EPOCH"],
            "--beta",
            env["BETA"],
            "--latent_size",
            env["LATENT_SIZE"],
            "--grid_x",
            env["GRID_X"],
            "--grid_y",
            env["GRID_Y"],
            "--grid_z",
            env["GRID_Z"],
            "--seed",
            env["SEED"],
            "--base_data_path",
            env["DATA_DIR"],
            "--result_base_path",
            env["RESULT_DIR"],
        ],
        check=True,
    )

    progress.set_postfix(iter=iter_idx, stage="generate")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "orr_vae.cli.main",
            "generate-structures",
            "--iter",
            str(iter_idx),
            "--num",
            env["GENERATED_NUM_STRUCTURES"],
            "--overpotential_condition",
            "1",
            "--alloy_stability_condition",
            "1",
            "--latent_size",
            env["LATENT_SIZE"],
            "--grid_x",
            env["GRID_X"],
            "--grid_y",
            env["GRID_Y"],
            "--grid_z",
            env["GRID_Z"],
            "--seed",
            env["SEED"],
            "--output_dir",
            env["DATA_DIR"],
            "--result_dir",
            env["RESULT_DIR"],
        ],
        check=True,
    )

    if with_visualization:
        progress.set_postfix(iter=iter_idx, stage="visualize")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "orr_vae.cli.main",
                "visualize-latent",
                "--iter",
                str(iter_idx),
                "--latent_size",
                env["LATENT_SIZE"],
                "--grid_x",
                env["GRID_X"],
                "--grid_y",
                env["GRID_Y"],
                "--grid_z",
                env["GRID_Z"],
                "--seed",
                env["SEED"],
                "--data_dir",
                env["DATA_DIR"],
                "--result_dir",
                env["RESULT_DIR"],
            ],
            check=True,
        )
PY

if [ "${WITH_ANALYSIS}" = "1" ]; then
  python3 -m orr_vae.cli.main analyze \
    --iter "${MAX_ITER}" \
    --base_path "${DATA_DIR}" \
    --output_path "${RESULT_DIR}/figure"
fi

echo "[Pt-Ni] completed: ${OUTPUT_DIR}"
