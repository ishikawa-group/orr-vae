"""Top-level CLI for ORR-VAE workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orr-vae", description="ORR catalyst screening workflows")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("calc-orr", help="Run ORR/formation-energy calculation helpers")
    sub.add_parser("train-cvae", help="Run conditional VAE training")
    sub.add_parser("generate-structures", help="Generate structures from trained CVAE")
    sub.add_parser("visualize-latent", help="Visualize latent space")
    sub.add_parser("analyze", help="Analyze ORR catalyst data")

    return parser


def _dispatch(module: str, argv: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *argv]
    return subprocess.call(cmd)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 0

    if argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0

    command = argv[0]
    rest = argv[1:]

    mapping = {
        "calc-orr": "orr_vae.workflows.calculate_overpotentials",
        "train-cvae": "orr_vae.workflows.conditional_vae",
        "generate-structures": "orr_vae.workflows.generate_new_structures",
        "visualize-latent": "orr_vae.workflows.visualize_latent_space",
        "analyze": "orr_vae.workflows.analyze_orr_catalyst_data",
    }

    if command not in mapping:
        build_parser().print_help()
        return 2

    return _dispatch(mapping[command], rest)


if __name__ == "__main__":
    raise SystemExit(main())
