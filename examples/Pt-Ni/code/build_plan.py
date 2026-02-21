#!/usr/bin/env python3
"""Emit shell exports derived from Pt-Ni settings."""

from __future__ import annotations

import argparse
import shlex

from settings import load_settings


def _sh(key: str, value: str) -> str:
    return f"export {key}={shlex.quote(str(value))}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build shell exports for Pt-Ni workflow")
    parser.add_argument("--shell", action="store_true", help="Emit shell export lines (default)")
    _ = parser.parse_args()

    cfg = load_settings()

    exports = {
        "SEED": cfg.workflow.seed,
        "LABEL_THRESHOLD": cfg.workflow.label_threshold,
        "BATCH_SIZE": cfg.workflow.batch_size,
        "MAX_EPOCH": cfg.workflow.max_epoch,
        "LATENT_SIZE": cfg.workflow.latent_size,
        "BETA": cfg.workflow.beta,
        "MAX_ITER": cfg.workflow.max_iter,
        "CALCULATOR": cfg.workflow.calculator,
        "JOB_NUM": cfg.workflow.job_num,
        "OUTPUT_DIR": cfg.workflow.output_dir,
        "DATA_DIR": cfg.workflow.data_dir,
        "RESULT_DIR": cfg.workflow.result_dir,
        "LOG_DIR": cfg.workflow.log_dir,
        "SOLVENT_CORRECTION_YAML": cfg.workflow.solvent_correction_yaml,
        "INITIAL_NUM_STRUCTURES": cfg.generation.initial_num_structures,
        "GENERATED_NUM_STRUCTURES": cfg.generation.generated_num_structures,
        "GRID_X": cfg.generation.size[0],
        "GRID_Y": cfg.generation.size[1],
        "GRID_Z": cfg.generation.size[2],
        "WITH_VISUALIZATION": int(cfg.workflow.with_visualization),
        "WITH_ANALYSIS": int(cfg.workflow.with_analysis),
        "MIN_SECONDARY_FRACTION": cfg.generation.min_fraction_secondary,
        "MAX_SECONDARY_FRACTION": cfg.generation.max_fraction_secondary,
    }

    for k, v in exports.items():
        print(_sh(k, str(v)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
