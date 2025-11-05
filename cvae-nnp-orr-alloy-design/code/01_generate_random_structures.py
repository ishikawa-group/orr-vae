#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
import numpy as np
from tool import vegard_lattice_constant

# ------------------------------
# Argument parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Generate random Pt-Ni alloy slabs on fcc(111)")
parser.add_argument("--num", type=int, default=128,
                    help="Number of structures to generate (default: 128)")
parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "data"),  
                    help="Output directory (default: ./data)")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed (default: 0)")
args = parser.parse_args()

# ------------------------------
# Global configuration
# ------------------------------
size = [4, 4, 4]
vacuum = None
alloy_elements = ["Pt", "Ni"]

# ------------------------------
# Prepare output directory
# ------------------------------
data_dir = args.output_dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

db_path = os.path.join(data_dir, "iter0_structures.json")
db = connect(db_path)

np.random.seed(args.seed)

print(f"Generating {args.num} random alloy structures...")

# ------------------------------
# Structure generation loop
# ------------------------------
for i in range(args.num):
    ni_fraction = np.random.uniform(1 / 64, 63 / 64)
    pt_fraction = 1.0 - ni_fraction
    fractions = [pt_fraction, ni_fraction]
    
    lattice_const = vegard_lattice_constant(alloy_elements, fractions)
    
    bulk = fcc111(symbol="Pt", 
                  size=size, 
                  a=lattice_const,
                  vacuum=vacuum, 
                  periodic=True)

    natoms = len(bulk)
    n_ni = int(round(natoms * ni_fraction))
    n_pt = natoms - n_ni

    alloy_list = ([atomic_numbers["Pt"]] * n_pt + 
                  [atomic_numbers["Ni"]] * n_ni)

    np.random.shuffle(alloy_list)
    bulk.set_atomic_numbers(alloy_list)

    bulk.calc = EMT()

    ads_info = bulk.info["adsorbate_info"]

    data = {
        "chemical_formula": bulk.get_chemical_formula(), 
        "ni_fraction": float(ni_fraction),
        "pt_fraction": float(pt_fraction),
        "lattice_constant": float(lattice_const),
        "run": i,
        "adsorbate_info": ads_info
    }
    
    db.write(bulk, data=data)
    
    if (i+1) % 10 == 0:
        print(f"Generated {i+1}/{args.num} structures")

print(f"Structures saved to {db_path}")
