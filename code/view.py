#!/usr/bin/env python3

from ase.io import read
from ase.visualize import view
from ase import Atoms

# Path to trajectory file
file = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/matter_sim/OH/adsorption/ofst_3.0_3.xyz"

# Read the trajectory file
atoms = read(
            file, 
             #index=':'
             )  # ':' reads all frames

#atoms.wrap()

# Visualize the trajectory
view(atoms)