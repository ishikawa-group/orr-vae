from ase.build import fcc111, add_adsorbate
from ase import Atoms
from ase.io import write
from ase.visualize import view

# Create copper surface
surf = fcc111("Pt", size=[4,4,3], vacuum=10.0)

# Create CO molecule
mol = Atoms("CO")

# Add CO to the surface at an ontop position
add_adsorbate(surf, mol, 1.6, position="ontop")

# Save the structure as an image
#write('slab.png', surf)
view(mol)
