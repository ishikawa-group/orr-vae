#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../code" && pwd)"

if [ -z "${JOB_NUM:-}" ]; then
  echo "JOB_NUM is required" >&2
  exit 1
fi

export CONDITION_FILE="${CONDITION_FILE:-${CODE_DIR}/condition_list.csv}"

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

bash "${SCRIPT_DIR}/run_tsubame_vae.sh"
