#!/usr/bin/env python3
"""Initial random dataset generation for Pt-Ni / Pt-Ti / Pt-Y example."""

from __future__ import annotations

import argparse
import os

import numpy as np
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.data import atomic_numbers
from ase.db import connect

from orr_vae.tool import vegard_lattice_constant
from settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate iter0 random slabs for Pt-Ni / Pt-Ti / Pt-Y binary systems")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for iter0_structures.json")
    parser.add_argument("--num", type=int, default=None, help="Number of initial structures")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_settings()

    output_dir = args.output_dir or str(cfg.workflow.data_dir)
    total_num = int(args.num if args.num is not None else cfg.generation.initial_num_structures)
    seed = int(args.seed if args.seed is not None else cfg.workflow.seed)

    size = cfg.generation.size
    vacuum = cfg.generation.vacuum
    variants = cfg.generation.binary_variants
    all_elements = cfg.generation.all_elements
    natoms = int(np.prod(size))

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "iter0_structures.json")
    db = connect(db_path)

    np.random.seed(seed)

    num_variants = len(variants)
    base_per_variant = total_num // num_variants
    remainder = total_num % num_variants
    structures_per_variant = [base_per_variant] * num_variants
    for idx in range(remainder):
        structures_per_variant[idx] += 1

    print(f"[Pt-Ni_Pt-Ti_Pt-Y] Generating initial dataset: {total_num} structures")
    print(f"[Pt-Ni_Pt-Ti_Pt-Y] variants={variants}, size={size}, vacuum={vacuum}")

    generated = 0
    for variant_idx, elements in enumerate(variants):
        target_count = structures_per_variant[variant_idx]
        if target_count == 0:
            continue

        print(f"  variant {elements}: target {target_count}")
        for _ in range(target_count):
            while True:
                fractions = np.random.dirichlet(np.ones(len(elements)))
                counts = np.random.multinomial(natoms, fractions)
                if np.all(counts > 0):
                    break
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

    print(f"[Pt-Ni_Pt-Ti_Pt-Y] saved to {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
