#!/usr/bin/env python3

from ase.io import read
from ase.visualize import view
from ase import Atoms

# Path to trajectory file
file = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/temp_f77cfe833cb8481da42342e9eb71078f/f77cfe833cb8481da42342e9eb71078f/OH/adsorption/ofst_2.5_2.xyz"

# Read the trajectory file
atoms = read(
            file, 
             #index=':'
             )  # ':' reads all frames

#atoms.wrap()

# Visualize the trajectory
view(atoms)