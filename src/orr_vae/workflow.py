"""High-level workflow API for ORR-VAE iterative screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional
import os
import subprocess
import sys

from tqdm.auto import tqdm


InitialGenerator = Callable[["WorkflowConfig"], None]


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for running the full evaluate/train/generate loop."""

    example_name: str
    seed: int
    label_threshold: float
    batch_size: int
    max_epoch: int
    latent_size: int
    beta: float
    max_iter: int
    calculator: str
    with_visualization: bool
    with_analysis: bool
    initial_num_structures: int
    generated_num_structures: int
    grid_x: int
    grid_y: int
    grid_z: int
    output_dir: Path
    data_dir: Path
    result_dir: Path
    log_dir: Path
    temp_dir: Path
    solvent_correction_yaml: Path
    initial_generator: InitialGenerator
    overpotential_condition: int = 1
    alloy_stability_condition: int = 1
    keep_temp_outputs: bool = True
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    """Summary of a completed workflow execution."""

    success: bool
    completed_iters: int
    output_dir: Path
    data_dir: Path
    result_dir: Path
    artifacts: Dict[str, str]


def _run(cmd: list[str], *, env: Dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def run_workflow(config: WorkflowConfig) -> WorkflowResult:
    """Run initial generation and iterative ORR/CVAE workflow."""
    env = os.environ.copy()
    env.update(config.environment)

    output_dir = Path(config.output_dir)
    data_dir = Path(config.data_dir)
    result_dir = Path(config.result_dir)
    log_dir = Path(config.log_dir)
    temp_dir = Path(config.temp_dir)
    solvent_yaml = Path(config.solvent_correction_yaml)

    for directory in (output_dir, data_dir, result_dir, log_dir, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"[{config.example_name}] output_dir={output_dir}")
    print(f"[{config.example_name}] data_dir={data_dir}")
    print(f"[{config.example_name}] result_dir={result_dir}")
    print(f"[{config.example_name}] keep_temp_outputs={config.keep_temp_outputs}")

    config.initial_generator(config)

    progress = tqdm(
        range(config.max_iter + 1),
        desc=f"{config.example_name} workflow",
        unit="iter",
        dynamic_ncols=True,
        disable=False,
    )

    completed_iters = -1
    for iter_idx in progress:
        print(
            f"[{config.example_name}] ===== iter {iter_idx}/{config.max_iter} : "
            "evaluate -> train -> generate =====",
            flush=True,
        )

        progress.set_postfix(iter=iter_idx, stage="evaluate")
        calc_orr_cmd = [
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
            config.calculator,
            "--solvent_correction_yaml_path",
            str(solvent_yaml),
        ]
        if config.keep_temp_outputs:
            calc_orr_cmd.append("--keep_temp")
        _run(calc_orr_cmd, env=env)

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
                str(config.label_threshold),
                "--batch_size",
                str(config.batch_size),
                "--max_epoch",
                str(config.max_epoch),
                "--beta",
                str(config.beta),
                "--latent_size",
                str(config.latent_size),
                "--grid_x",
                str(config.grid_x),
                "--grid_y",
                str(config.grid_y),
                "--grid_z",
                str(config.grid_z),
                "--seed",
                str(config.seed),
                "--base_data_path",
                str(data_dir),
                "--result_base_path",
                str(result_dir),
            ],
            env=env,
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
                str(config.generated_num_structures),
                "--overpotential_condition",
                str(config.overpotential_condition),
                "--alloy_stability_condition",
                str(config.alloy_stability_condition),
                "--latent_size",
                str(config.latent_size),
                "--grid_x",
                str(config.grid_x),
                "--grid_y",
                str(config.grid_y),
                "--grid_z",
                str(config.grid_z),
                "--seed",
                str(config.seed),
                "--output_dir",
                str(data_dir),
                "--result_dir",
                str(result_dir),
            ],
            env=env,
        )

        if config.with_visualization:
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
                    str(config.latent_size),
                    "--grid_x",
                    str(config.grid_x),
                    "--grid_y",
                    str(config.grid_y),
                    "--grid_z",
                    str(config.grid_z),
                    "--seed",
                    str(config.seed),
                    "--data_dir",
                    str(data_dir),
                    "--result_dir",
                    str(result_dir),
                ],
                env=env,
            )

        completed_iters = iter_idx

    if config.with_analysis:
        _run(
            [
                sys.executable,
                "-m",
                "orr_vae.cli.main",
                "analyze",
                "--iter",
                str(config.max_iter),
                "--base_path",
                str(data_dir),
                "--output_path",
                str(result_dir / "figure"),
            ],
            env=env,
        )

    artifacts = {
        "result_dir": str(result_dir),
        "data_dir": str(data_dir),
        "latest_model": str(result_dir / f"iter{config.max_iter}" / f"final_cvae_iter{config.max_iter}.pt"),
        "latest_structures": str(data_dir / f"iter{config.max_iter + 1}_structures.json"),
        "latest_results": str(data_dir / f"iter{config.max_iter}_calculation_result.json"),
    }

    print(f"[{config.example_name}] completed: {output_dir}")
    return WorkflowResult(
        success=True,
        completed_iters=completed_iters,
        output_dir=output_dir,
        data_dir=data_dir,
        result_dir=result_dir,
        artifacts=artifacts,
    )
