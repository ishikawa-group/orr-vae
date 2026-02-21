#!/usr/bin/env python3
"""Single-entry workflow runner for the Pt-Ni example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.data import atomic_numbers
from ase.db import connect


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orr_vae import WorkflowConfig, run_workflow, vegard_lattice_constant


def _bool_int(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pt-Ni ORR-VAE workflow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--job-num", default="manual")
    parser.add_argument("--label-threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--calculator", default="fairchem")
    parser.add_argument("--with-visualization", type=_bool_int, default=True)
    parser.add_argument("--with-analysis", type=_bool_int, default=True)
    parser.add_argument("--grid-x", type=int, default=4)
    parser.add_argument("--grid-y", type=int, default=4)
    parser.add_argument("--grid-z", type=int, default=6)
    parser.add_argument("--vacuum", default="none")
    parser.add_argument("--initial-num-structures", type=int, default=128)
    parser.add_argument("--generated-num-structures", type=int, default=128)
    parser.add_argument("--elements", default="Pt,Ni")
    parser.add_argument("--min-secondary-fraction", type=float, default=1.0 / 96.0)
    parser.add_argument("--max-secondary-fraction", type=float, default=95.0 / 96.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--solvent-correction-yaml", type=Path, default=None)
    parser.add_argument("--overpotential-condition", type=int, choices=[0, 1], default=1)
    parser.add_argument("--alloy-stability-condition", type=int, choices=[0, 1], default=1)
    return parser


def _parse_elements(raw: str) -> tuple[str, str]:
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(items) != 2:
        raise ValueError(f"--elements must contain exactly two symbols, got: {raw}")
    return items[0], items[1]


def _parse_vacuum(raw: str) -> float | None:
    lowered = raw.strip().lower()
    if lowered in {"none", ""}:
        return None
    return float(raw)


def _build_initial_generator(
    *,
    elements: tuple[str, str],
    size: tuple[int, int, int],
    vacuum: float | None,
    min_secondary_fraction: float,
    max_secondary_fraction: float,
) -> callable:
    def _generate(config: WorkflowConfig) -> None:
        np.random.seed(config.seed)

        data_dir = Path(config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "iter0_structures.json"
        db = connect(str(db_path))

        print(f"[{config.example_name}] Generating initial dataset: {config.initial_num_structures} structures")
        print(f"[{config.example_name}] size={size}, vacuum={vacuum}, elements={elements}")

        for i in range(config.initial_num_structures):
            sec_frac = np.random.uniform(min_secondary_fraction, max_secondary_fraction)
            pri_frac = 1.0 - sec_frac
            fractions = [pri_frac, sec_frac]

            lattice_const = vegard_lattice_constant(list(elements), fractions)
            bulk = fcc111(symbol=elements[0], size=size, a=lattice_const, vacuum=vacuum, periodic=True)

            natoms = len(bulk)
            n_secondary = int(round(natoms * sec_frac))
            n_primary = natoms - n_secondary

            alloy_list = [atomic_numbers[elements[0]]] * n_primary + [atomic_numbers[elements[1]]] * n_secondary
            np.random.shuffle(alloy_list)
            bulk.set_atomic_numbers(alloy_list)
            bulk.calc = EMT()

            ads_info = bulk.info.get("adsorbate_info", {})
            data = {
                "chemical_formula": bulk.get_chemical_formula(),
                f"{elements[0].lower()}_fraction": float(pri_frac),
                f"{elements[1].lower()}_fraction": float(sec_frac),
                "lattice_constant": float(lattice_const),
                "run": i,
                "adsorbate_info": ads_info,
                "binary_alloy": f"{elements[0]}-{elements[1]}",
            }
            db.write(bulk, data=data)

            if (i + 1) % 10 == 0 or (i + 1) == config.initial_num_structures:
                print(f"  generated {i + 1}/{config.initial_num_structures}")

        print(f"[{config.example_name}] saved to {db_path}")

    return _generate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    example_dir = Path(__file__).resolve().parents[1]
    code_dir = Path(__file__).resolve().parent

    output_dir = (args.output_dir or (example_dir / "results" / f"seed_{args.seed}" / args.job_num)).resolve()
    data_dir = (args.data_dir or (output_dir / "data")).resolve()
    result_dir = (args.result_dir or (output_dir / "result")).resolve()
    log_dir = (args.log_dir or (result_dir / "log")).resolve()
    temp_dir = (args.temp_dir or (output_dir / "temp")).resolve()
    solvent_yaml = (args.solvent_correction_yaml or (code_dir / "solvent_correction.yaml")).resolve()

    os.environ["PYTHONPATH"] = f"{SRC_DIR}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")

    elements = _parse_elements(args.elements)
    vacuum = _parse_vacuum(args.vacuum)
    size = (args.grid_x, args.grid_y, args.grid_z)

    config = WorkflowConfig(
        example_name="Pt-Ni",
        seed=args.seed,
        label_threshold=args.label_threshold,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        latent_size=args.latent_size,
        beta=args.beta,
        max_iter=args.max_iter,
        calculator=args.calculator,
        with_visualization=bool(args.with_visualization),
        with_analysis=bool(args.with_analysis),
        initial_num_structures=args.initial_num_structures,
        generated_num_structures=args.generated_num_structures,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        grid_z=args.grid_z,
        output_dir=output_dir,
        data_dir=data_dir,
        result_dir=result_dir,
        log_dir=log_dir,
        temp_dir=temp_dir,
        solvent_correction_yaml=solvent_yaml,
        initial_generator=_build_initial_generator(
            elements=elements,
            size=size,
            vacuum=vacuum,
            min_secondary_fraction=args.min_secondary_fraction,
            max_secondary_fraction=args.max_secondary_fraction,
        ),
        overpotential_condition=args.overpotential_condition,
        alloy_stability_condition=args.alloy_stability_condition,
    )

    result = run_workflow(config)
    print(f"[Pt-Ni] workflow result: success={result.success}, completed_iters={result.completed_iters}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
