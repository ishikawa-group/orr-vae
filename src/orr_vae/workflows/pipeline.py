"""Iterative ORR screening pipeline runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run iterative ORR screening pipeline")
    parser.add_argument("--data_dir", type=str, default=str(Path.cwd() / "data"))
    parser.add_argument("--result_dir", type=str, default=str(Path.cwd() / "result"))
    parser.add_argument("--output_dir", type=str, default=str(Path.cwd()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_structures", type=int, default=128)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--calculator", type=str, default="fairchem")
    parser.add_argument("--with_visualization", action="store_true")
    parser.add_argument("--with_analysis", action="store_true")
    return parser


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[pipeline]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    python = sys.executable

    _run(
        [
            python,
            "-m",
            "orr_vae.workflows.generate_random_structures",
            "--num",
            str(args.num_structures),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(args.data_dir),
        ]
    )

    for iter_idx in range(args.max_iter + 1):
        _run(
            [
                python,
                "-m",
                "orr_vae.workflows.calculate_overpotentials",
                "run-all",
                "--iter",
                str(iter_idx),
                "--base_dir",
                str(args.output_dir),
                "--base_data_dir",
                str(args.data_dir),
                "--temp_base_dir",
                str(Path(args.output_dir) / "temp"),
                "--calculator",
                args.calculator,
            ]
        )
        _run(
            [
                python,
                "-m",
                "orr_vae.workflows.conditional_vae",
                "--iter",
                str(iter_idx),
                "--label_threshold",
                str(args.label_threshold),
                "--batch_size",
                str(args.batch_size),
                "--max_epoch",
                str(args.max_epoch),
                "--beta",
                str(args.beta),
                "--latent_size",
                str(args.latent_size),
                "--seed",
                str(args.seed),
                "--base_data_path",
                str(args.data_dir),
                "--result_base_path",
                str(args.result_dir),
            ]
        )
        _run(
            [
                python,
                "-m",
                "orr_vae.workflows.generate_new_structures",
                "--iter",
                str(iter_idx),
                "--num",
                str(args.num_structures),
                "--overpotential_condition",
                "1",
                "--alloy_stability_condition",
                "1",
                "--latent_size",
                str(args.latent_size),
                "--seed",
                str(args.seed),
                "--output_dir",
                str(args.data_dir),
                "--result_dir",
                str(args.result_dir),
            ]
        )
        if args.with_visualization:
            _run(
                [
                    python,
                    "-m",
                    "orr_vae.workflows.visualize_latent_space",
                    "--iter",
                    str(iter_idx),
                    "--latent_size",
                    str(args.latent_size),
                    "--seed",
                    str(args.seed),
                    "--data_dir",
                    str(args.data_dir),
                    "--result_dir",
                    str(args.result_dir),
                ]
            )

    if args.with_analysis:
        _run(
            [
                python,
                "-m",
                "orr_vae.workflows.analyze_orr_catalyst_data",
                "--iter",
                str(args.max_iter),
                "--base_path",
                str(args.data_dir),
                "--output_path",
                str(Path(args.result_dir) / "figure"),
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
