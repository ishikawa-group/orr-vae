#!/usr/bin/env python3
"""Run the full Pt-Ni iterative screening workflow."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_plan_env(code_dir: Path) -> None:
    output = subprocess.check_output(
        [sys.executable, str(code_dir / "build_plan.py"), "--shell"],
        text=True,
    )
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("export "):
            raise ValueError(f"Unexpected build_plan output: {line}")
        key, raw_value = line[len("export ") :].split("=", 1)
        parsed = shlex.split(raw_value)
        value = parsed[0] if parsed else ""
        os.environ[key] = value


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    example_dir = code_dir.parent
    root_dir = example_dir.parents[1]
    example_name = example_dir.name

    os.environ["PYTHONPATH"] = f"{root_dir / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    _load_plan_env(code_dir)

    data_dir = Path(os.environ["DATA_DIR"])
    result_dir = Path(os.environ["RESULT_DIR"])
    log_dir = Path(os.environ["LOG_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    temp_dir = output_dir / "temp"
    for directory in (data_dir, result_dir, log_dir, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"[{example_name}] output_dir={output_dir}")
    print(f"[{example_name}] data_dir={data_dir}")
    print(f"[{example_name}] result_dir={result_dir}")
    binary_variants = os.environ.get("BINARY_VARIANTS")
    if binary_variants:
        print(f"[{example_name}] binary variants: {binary_variants}")

    _run(
        [
            sys.executable,
            str(code_dir / "initial_generation.py"),
            "--output_dir",
            str(data_dir),
            "--num",
            os.environ["INITIAL_NUM_STRUCTURES"],
            "--seed",
            os.environ["SEED"],
        ]
    )

    max_iter = int(os.environ["MAX_ITER"])
    with_visualization = os.environ.get("WITH_VISUALIZATION", "0") == "1"
    progress = tqdm(
        range(max_iter + 1),
        desc=f"{example_name} workflow",
        unit="iter",
        dynamic_ncols=True,
        disable=False,
    )

    for iter_idx in progress:
        print(
            f"[{example_name}] ===== iter {iter_idx}/{max_iter} : evaluate -> train -> generate =====",
            flush=True,
        )

        progress.set_postfix(iter=iter_idx, stage="evaluate")
        _run(
            [
                sys.executable,
                "-m",
                "orr_vae.cli.main",
                "calc-orr",
                "run-all",
                "--iter",
                str(iter_idx),
                "--base_dir",
                str(output_dir),
                "--base_data_dir",
                str(data_dir),
                "--temp_base_dir",
                str(temp_dir),
                "--calculator",
                os.environ["CALCULATOR"],
                "--solvent_correction_yaml_path",
                os.environ["SOLVENT_CORRECTION_YAML"],
            ]
        )

        progress.set_postfix(iter=iter_idx, stage="train")
        _run(
            [
                sys.executable,
                "-m",
                "orr_vae.cli.main",
                "train-cvae",
                "--iter",
                str(iter_idx),
                "--label_threshold",
                os.environ["LABEL_THRESHOLD"],
                "--batch_size",
                os.environ["BATCH_SIZE"],
                "--max_epoch",
                os.environ["MAX_EPOCH"],
                "--beta",
                os.environ["BETA"],
                "--latent_size",
                os.environ["LATENT_SIZE"],
                "--grid_x",
                os.environ["GRID_X"],
                "--grid_y",
                os.environ["GRID_Y"],
                "--grid_z",
                os.environ["GRID_Z"],
                "--seed",
                os.environ["SEED"],
                "--base_data_path",
                str(data_dir),
                "--result_base_path",
                str(result_dir),
            ]
        )

        progress.set_postfix(iter=iter_idx, stage="generate")
        _run(
            [
                sys.executable,
                "-m",
                "orr_vae.cli.main",
                "generate-structures",
                "--iter",
                str(iter_idx),
                "--num",
                os.environ["GENERATED_NUM_STRUCTURES"],
                "--overpotential_condition",
                "1",
                "--alloy_stability_condition",
                "1",
                "--latent_size",
                os.environ["LATENT_SIZE"],
                "--grid_x",
                os.environ["GRID_X"],
                "--grid_y",
                os.environ["GRID_Y"],
                "--grid_z",
                os.environ["GRID_Z"],
                "--seed",
                os.environ["SEED"],
                "--output_dir",
                str(data_dir),
                "--result_dir",
                str(result_dir),
            ]
        )

        if with_visualization:
            progress.set_postfix(iter=iter_idx, stage="visualize")
            _run(
                [
                    sys.executable,
                    "-m",
                    "orr_vae.cli.main",
                    "visualize-latent",
                    "--iter",
                    str(iter_idx),
                    "--latent_size",
                    os.environ["LATENT_SIZE"],
                    "--grid_x",
                    os.environ["GRID_X"],
                    "--grid_y",
                    os.environ["GRID_Y"],
                    "--grid_z",
                    os.environ["GRID_Z"],
                    "--seed",
                    os.environ["SEED"],
                    "--data_dir",
                    str(data_dir),
                    "--result_dir",
                    str(result_dir),
                ]
            )

    if os.environ.get("WITH_ANALYSIS", "0") == "1":
        _run(
            [
                sys.executable,
                "-m",
                "orr_vae.cli.main",
                "analyze",
                "--iter",
                str(max_iter),
                "--base_path",
                str(data_dir),
                "--output_path",
                str(result_dir / "figure"),
            ]
        )

    print(f"[{example_name}] completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
