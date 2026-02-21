#!/usr/bin/env python3
"""Single-entry workflow runner for the Pt-Ni / Pt-Ti / Pt-Y example."""

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
    parser = argparse.ArgumentParser(description="Run Pt-Ni_Pt-Ti_Pt-Y ORR-VAE workflow")
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
    parser.add_argument("--keep-temp", type=_bool_int, default=True)
    parser.add_argument("--grid-x", type=int, default=4)
    parser.add_argument("--grid-y", type=int, default=4)
    parser.add_argument("--grid-z", type=int, default=6)
    parser.add_argument("--vacuum", default="none")
    parser.add_argument("--initial-num-structures", type=int, default=192)
    parser.add_argument("--generated-num-structures", type=int, default=192)
    parser.add_argument("--binary-variants", default="Pt-Ni,Pt-Ti,Pt-Y")
    parser.add_argument("--all-elements", default="Pt,Ni,Ti,Y")
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


def _parse_vacuum(raw: str) -> float | None:
    lowered = raw.strip().lower()
    if lowered in {"none", ""}:
        return None
    return float(raw)


def _parse_binary_variants(raw: str) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        elems = [item.strip() for item in token.split("-") if item.strip()]
        if len(elems) != 2:
            raise ValueError(f"Invalid binary variant token: {token}")
        variants.append((elems[0], elems[1]))
    if not variants:
        raise ValueError("--binary-variants cannot be empty")
    return tuple(variants)


def _parse_all_elements(raw: str) -> tuple[str, ...]:
    elems = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not elems:
        raise ValueError("--all-elements cannot be empty")
    return elems


def _build_initial_generator(
    *,
    size: tuple[int, int, int],
    vacuum: float | None,
    binary_variants: tuple[tuple[str, str], ...],
    all_elements: tuple[str, ...],
    min_secondary_fraction: float,
    max_secondary_fraction: float,
) -> callable:
    def _generate(config: WorkflowConfig) -> None:
        np.random.seed(config.seed)

        data_dir = Path(config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "iter0_structures.json"
        db = connect(str(db_path))

        natoms = int(np.prod(size))
        min_secondary_count = max(1, int(np.ceil(natoms * min_secondary_fraction)))
        max_secondary_count = min(natoms - 1, int(np.floor(natoms * max_secondary_fraction)))
        if min_secondary_count > max_secondary_count:
            raise ValueError(
                f"Invalid secondary fraction range: {min_secondary_fraction} - {max_secondary_fraction} "
                f"for natoms={natoms}"
            )

        num_variants = len(binary_variants)
        total_num = config.initial_num_structures
        base_per_variant = total_num // num_variants
        remainder = total_num % num_variants
        structures_per_variant = [base_per_variant] * num_variants
        for idx in range(remainder):
            structures_per_variant[idx] += 1

        print(f"[{config.example_name}] Generating initial dataset: {total_num} structures")
        print(f"[{config.example_name}] variants={binary_variants}, size={size}, vacuum={vacuum}")
        print(
            f"[{config.example_name}] secondary count range per slab: "
            f"{min_secondary_count}..{max_secondary_count} / {natoms}"
        )

        generated = 0
        for variant_idx, elements in enumerate(binary_variants):
            target_count = structures_per_variant[variant_idx]
            if target_count == 0:
                continue

            print(f"  variant {elements}: target {target_count}")
            for _ in range(target_count):
                n_secondary = int(np.random.randint(min_secondary_count, max_secondary_count + 1))
                n_primary = natoms - n_secondary
                counts = np.array([n_primary, n_secondary], dtype=int)
                fractions = counts / natoms

                lattice_const = vegard_lattice_constant(list(elements), fractions.tolist())
                bulk = fcc111(symbol="Pt", size=size, a=lattice_const, vacuum=vacuum, periodic=True)

                alloy_list: list[int] = []
                for element, count in zip(elements, counts):
                    alloy_list.extend([atomic_numbers[element]] * int(count))
                np.random.shuffle(alloy_list)

                bulk.set_atomic_numbers(alloy_list)
                bulk.calc = EMT()

                composition = {element: 0.0 for element in all_elements}
                for element, frac in zip(elements, fractions):
                    composition[element] = float(frac)

                data = {
                    "chemical_formula": bulk.get_chemical_formula(),
                    "composition": composition,
                    "lattice_constant": float(lattice_const),
                    "run": generated,
                    "adsorbate_info": bulk.info.get("adsorbate_info", {}),
                    "binary_alloy": f"{elements[0]}-{elements[1]}",
                }
                for element in all_elements:
                    data[f"{element.lower()}_fraction"] = float(composition.get(element, 0.0))

                db.write(bulk, data=data)
                generated += 1

                if generated % 10 == 0 or generated == total_num:
                    print(f"  generated {generated}/{total_num}")

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

    binary_variants = _parse_binary_variants(args.binary_variants)
    all_elements = _parse_all_elements(args.all_elements)
    size = (args.grid_x, args.grid_y, args.grid_z)
    vacuum = _parse_vacuum(args.vacuum)

    config = WorkflowConfig(
        example_name="Pt-Ni_Pt-Ti_Pt-Y",
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
            size=size,
            vacuum=vacuum,
            binary_variants=binary_variants,
            all_elements=all_elements,
            min_secondary_fraction=args.min_secondary_fraction,
            max_secondary_fraction=args.max_secondary_fraction,
        ),
        overpotential_condition=args.overpotential_condition,
        alloy_stability_condition=args.alloy_stability_condition,
        keep_temp_outputs=bool(args.keep_temp),
    )

    result = run_workflow(config)
    print(
        "[Pt-Ni_Pt-Ti_Pt-Y] workflow result: "
        f"success={result.success}, completed_iters={result.completed_iters}"
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
