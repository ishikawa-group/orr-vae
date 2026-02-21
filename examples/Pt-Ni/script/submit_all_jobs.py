#!/usr/bin/env python3
"""Submit many Pt-Ni jobs from condition_list.csv."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Pt-Ni jobs to qsub")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-temp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--group", default="tga-ishikawalab")
    parser.add_argument("--job-prefix", default="VAE_PtNi")
    parser.add_argument("--only", default="", help="Comma-separated job numbers")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_env(
    *,
    job_num: str,
    row: Dict[str, str],
    seed: int,
    keep_temp: int,
    root_dir: Path,
    example_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    data_dir = output_dir / "data"
    result_dir = output_dir / "result"
    log_dir = result_dir / "log"
    temp_dir = output_dir / "temp"

    env = os.environ.copy()
    env["JOB_NUM"] = job_num
    env["SEED"] = str(seed)
    env["LABEL_THRESHOLD"] = str(row["label_threshold"])
    env["BATCH_SIZE"] = str(row["batch_size"])
    env["MAX_EPOCH"] = str(row["max_epoch"])
    env["LATENT_SIZE"] = str(row["latent_size"])

    # Workflow defaults (single source of truth for submit jobs)
    env["BETA"] = "1.0"
    env["MAX_ITER"] = "5"
    env["CALCULATOR"] = "fairchem"
    env["WITH_VISUALIZATION"] = "1"
    env["WITH_ANALYSIS"] = "1"
    env["KEEP_TEMP"] = str(keep_temp)
    env["GRID_X"] = "4"
    env["GRID_Y"] = "4"
    env["GRID_Z"] = "6"
    env["INITIAL_NUM_STRUCTURES"] = "128"
    env["GENERATED_NUM_STRUCTURES"] = "128"
    env["MIN_SECONDARY_FRACTION"] = str(1.0 / 96.0)
    env["MAX_SECONDARY_FRACTION"] = str(95.0 / 96.0)

    # Runtime / path configuration
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["XDG_RUNTIME_DIR"] = f"/tmp/runtime-{os.getuid()}"
    env["MODULE_LOADS"] = "intel intel-mpi cuda"
    env["ORR_VAE_ROOT"] = str(root_dir)
    env["ORR_VAE_EXAMPLE_DIR"] = str(example_dir)
    env["ORR_VAE_CODE_DIR"] = str(example_dir / "code")
    env["VENV_PATH"] = str(root_dir / ".venv")
    env["OUTPUT_DIR"] = str(output_dir)
    env["DATA_DIR"] = str(data_dir)
    env["RESULT_DIR"] = str(result_dir)
    env["LOG_DIR"] = str(log_dir)
    env["TEMP_DIR"] = str(temp_dir)
    env["SOLVENT_CORRECTION_YAML"] = str(example_dir / "code" / "solvent_correction.yaml")

    return env


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    example_dir = script_dir.parent
    root_dir = example_dir.parents[1]
    condition_file = example_dir / "code" / "condition_list.csv"
    job_script = script_dir / "run_tsubame_vae_parametric.sh"

    if not condition_file.exists():
        raise FileNotFoundError(f"Condition file not found: {condition_file}")

    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    submitted = 0
    failed: list[str] = []

    with condition_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_num = str(row["n"])
            if selected and job_num not in selected:
                continue

            output_dir = example_dir / "results" / f"seed_{args.seed}" / job_num
            log_dir = output_dir / "result" / "log"
            log_dir.mkdir(parents=True, exist_ok=True)

            env = build_env(
                job_num=job_num,
                row=row,
                seed=args.seed,
                keep_temp=args.keep_temp,
                root_dir=root_dir,
                example_dir=example_dir,
                output_dir=output_dir,
            )

            cmd = [
                "qsub",
                "-V",
                "-g",
                args.group,
                "-o",
                str(log_dir / "output.log"),
                "-e",
                str(log_dir / "error.log"),
                "-N",
                f"{args.job_prefix}_{job_num}_seed{args.seed}",
                str(job_script),
            ]

            if args.dry_run:
                print("DRY-RUN:", " ".join(cmd))
                submitted += 1
                continue

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"submitted job {job_num}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(f"failed job {job_num}: {result.stderr.strip()}")
                failed.append(job_num)

    print(f"submitted: {submitted}")
    print(f"failed: {len(failed)}")
    if failed:
        print("failed jobs:", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
