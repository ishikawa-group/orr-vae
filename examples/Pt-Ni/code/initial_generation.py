#!/usr/bin/env python3
"""Initial random dataset generation for Pt-Ni example."""

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
    parser = argparse.ArgumentParser(description="Generate iter0 random Pt-Ni slabs")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for iter0_structures.json")
    parser.add_argument("--num", type=int, default=None, help="Number of initial structures")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_settings()

    size = cfg.generation.size
    vacuum = cfg.generation.vacuum
    elements = cfg.generation.elements

    output_dir = args.output_dir or str(cfg.workflow.data_dir)
    num = int(args.num if args.num is not None else cfg.generation.initial_num_structures)
    seed = int(args.seed if args.seed is not None else cfg.workflow.seed)

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "iter0_structures.json")
    db = connect(db_path)

    np.random.seed(seed)

    print(f"[Pt-Ni] Generating initial dataset: {num} structures")
    print(f"[Pt-Ni] size={size}, vacuum={vacuum}, elements={elements}")

    for i in range(num):
        sec_frac = np.random.uniform(
            cfg.generation.min_fraction_secondary,
            cfg.generation.max_fraction_secondary,
        )
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
        if (i + 1) % 10 == 0 or (i + 1) == num:
            print(f"  generated {i+1}/{num}")

    print(f"[Pt-Ni] saved to {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
