import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ase.db import connect
from ase.data import atomic_numbers
from typing import List, Union, Dict, Any


ALLOY_ELEMENTS = ["Pt", "Ni", "Ti", "Y"]
ALLOY_Z_NUMBERS = [atomic_numbers[element] for element in ALLOY_ELEMENTS]
Z_TO_ELEMENT = {z: element for element, z in zip(ALLOY_ELEMENTS, ALLOY_Z_NUMBERS)}
CLASS_TO_Z = {idx + 1: z for idx, z in enumerate(ALLOY_Z_NUMBERS)}
Z_TO_CLASS = {z: cls for cls, z in CLASS_TO_Z.items()}
VACANCY_CLASS = 0
NUM_CLASSES = len(ALLOY_ELEMENTS) + 1


def convert_numpy_types(obj):
    """Recursively convert NumPy scalars into native Python types."""
    import numpy as np
    if isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def atomic_numbers_tensor_to_classes(structure_tensor: torch.Tensor) -> torch.Tensor:
    """Convert atomic-number tensor into model class IDs (0: vacancy, 1..N: alloy elements)."""
    class_tensor = torch.zeros_like(structure_tensor, dtype=torch.long)
    for cls, z_num in CLASS_TO_Z.items():
        class_tensor[structure_tensor == z_num] = cls
    return class_tensor


def class_tensor_to_atomic_numbers(class_tensor: torch.Tensor) -> torch.Tensor:
    """Convert model class-ID tensor back into atomic numbers."""
    atomic_tensor = torch.zeros_like(class_tensor, dtype=torch.int64)
    for cls, z_num in CLASS_TO_Z.items():
        atomic_tensor[class_tensor == cls] = z_num
    return atomic_tensor


def normalize_composition(comp: Dict[str, float]) -> Dict[str, float]:
    """Fill missing elements and normalize fractions to 1.0 when possible."""
    result = {element: float(comp.get(element, 0.0)) for element in ALLOY_ELEMENTS}
    total = sum(result.values())
    if total <= 0:
        return {element: 0.0 for element in ALLOY_ELEMENTS}
    return {element: value / total for element, value in result.items()}


def compute_composition_from_structure(structure) -> Dict[str, float]:
    """Compute elemental fractions from an ASE Atoms structure."""
    symbols = structure.get_chemical_symbols()
    total = len(symbols)
    if total == 0:
        return {element: 0.0 for element in ALLOY_ELEMENTS}
    counts = {element: 0 for element in ALLOY_ELEMENTS}
    for symbol in symbols:
        if symbol in counts:
            counts[symbol] += 1
    return {element: counts[element] / total for element in ALLOY_ELEMENTS}


def extract_composition(entry: Dict[str, Any], structure=None) -> Dict[str, float]:
    """Extract normalized composition from JSON entry or fallback structure."""
    composition = entry.get("composition") or entry.get("element_fractions") or {}
    if not composition:
        composition = {element: float(entry.get(f"{element.lower()}_fraction", 0.0)) for element in ALLOY_ELEMENTS}
    composition = normalize_composition(composition)
    if sum(composition.values()) == 0 and structure is not None:
        composition = compute_composition_from_structure(structure)
    return composition


def elemental_a(symbol: str) -> float:
    """Return the FCC lattice constant (Å) for ``symbol`` from ASE reference states."""
    from ase.data import reference_states, atomic_numbers
    Z = atomic_numbers[symbol]
    a = reference_states[Z].get('a')
    if a is None:
        raise ValueError(f"No reference lattice constant for {symbol}")
    return a

def vegard_lattice_constant(elements, fractions=None):
    """
    Compute the Vegard-mixed lattice constant for the given elements.

    Parameters
    ----------
    elements : list[str]
        Element symbols such as ``['Pt', 'Ni']``.
    fractions : list[float] or None
        Fraction for each element. When ``None`` they are assumed to be uniform.
    """
    from ase.data import reference_states, atomic_numbers
    n = len(elements)
    if fractions is None:
        fractions = [1.0 / n] * n
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Fractions must sum to 1")
    constants = [elemental_a(el) for el in elements]
    return sum(a * x for a, x in zip(constants, fractions))

def get_number_of_layers(atoms):
    """
    Count the number of distinct layers based on the z coordinates.

    Parameters
    ----------
    atoms : ase.Atoms
        Target structure.

    Returns
    -------
    int
        Number of layers.
    """
    import numpy as np

    pos  = atoms.positions
    zpos = np.round(pos[:,2], decimals=3)
    nlayer = len(set(zpos))
    return nlayer


def set_tags_by_z(atoms):
    """
    Assign layer tags based on the z coordinate (0 for the bottom layer, 1 for the next, ...).
    """
    import numpy as np
    import pandas as pd

    newatoms = atoms.copy()
    pos = newatoms.positions
    # Round to one decimal place to determine layer thickness
    zpos = np.round(pos[:, 2], decimals=1)
    
    # Collect unique layer positions and sort in ascending order
    bins = np.sort(np.array(list(set(zpos)))) + 1.0e-2
    bins = np.insert(bins, 0, 0)
    
    # Map each bin to labels 0, 1, 2, ...
    labels = list(range(len(bins)-1))
    tags = pd.cut(zpos, bins=bins, labels=labels, include_lowest=True).tolist()
    newatoms.set_tags(tags)
    
    return newatoms

def sort_atoms(atoms, axes=("z", "y", "x")):
    """
    Sort an ``ase.Atoms`` object along the specified axes (default order: ``z``, ``y``, ``x``).

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to be sorted.
    axes : tuple[str, str, str]
        Sorting order, e.g. ``("z", "y", "x")``.

    Returns
    -------
    ase.Atoms
        Reordered structure.
    """
    import numpy as np
    
    axis_map = {"x": 0, "y": 1, "z": 2}
    pos = atoms.get_positions()  # shape: (n_atoms, 3)
    
    # ``np.lexsort`` gives priority to the last key, so reverse the axes order.
    keys = tuple(pos[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)
    
    sorted_atoms = atoms[sorted_indices]
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())
    
    return sorted_atoms


def structure_to_tensor(structure, grid_size):
    """
    Encode an fcc(111) slab into a 3D tensor following the ABC stacking scheme.

    Parameters
    ----------
    structure : ase.Atoms
        An fcc(111) slab whose atom count matches ``x * y * z``.
    grid_size : (int, int, int)
        Original cell counts ``[x, y, z]`` (``z`` should be divisible by 3 for full ABC cycles).

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(z, 2*y, 2*x)`` with atomic numbers mapped onto the ABC grid.
    """
    import torch
    x_size, y_size, z_size = grid_size
    if len(structure) != x_size * y_size * z_size:
        raise ValueError("Atom count does not match the provided grid_size")

    # Reorder atoms as (z, y, x) and reshape into a 3D array
    sorted_atoms = sort_atoms(structure, axes=("z", "y", "x"))
    basic = torch.tensor(sorted_atoms.get_atomic_numbers(),
                         dtype=torch.int64).reshape(z_size, y_size, x_size)

    # Initialise output tensor (0 represents vacancy)
    interleaved = torch.zeros((z_size, 2*y_size, 2*x_size), dtype=torch.int64)

    for z in range(z_size):
        layer = basic[z]
        mod = z % 3            # 0:A, 1:B, 2:C
        if mod == 0:           # A layer: even rows, even columns
            interleaved[z, 0::2, 0::2] = layer
        elif mod == 1:         # B layer: odd rows, odd columns
            interleaved[z, 1::2, 1::2] = layer
        else:                  # C layer: even rows, odd columns
            interleaved[z, 0::2, 1::2] = layer

    return interleaved



def tensor_to_structure(tensor, template_structure):
    """
    Reconstruct an ``ase.Atoms`` object from the interleaved tensor representation.

    Parameters
    ----------
    tensor : torch.Tensor
        Integer tensor of shape ``(z, 2*y, 2*x)`` generated by :func:`structure_to_tensor`.
    template_structure : ase.Atoms
        Structure with the desired positions/cell/PBC, already sorted in ``z-y-x`` order.

    Returns
    -------
    ase.Atoms
        New structure with atomic numbers replaced according to ``tensor``.
    """
    import torch
    z_size, ny, nx = tensor.shape
    y_size, x_size = ny // 2, nx // 2
    if len(template_structure) != z_size * y_size * x_size:
        raise ValueError("Atom count mismatch between tensor and template structure")

    layers = []
    for z in range(z_size):
        mod = z % 3
        if mod == 0:        # A layer occupies even rows/columns
            layer = tensor[z, 0::2, 0::2]
        elif mod == 1:      # B layer occupies odd rows/columns
            layer = tensor[z, 1::2, 1::2]
        else:               # C layer occupies even rows, odd columns
            layer = tensor[z, 0::2, 1::2]
        layers.append(layer.flatten())

    atomic_nums = torch.cat(layers).numpy()
    new_structure = template_structure.copy()
    new_structure.set_atomic_numbers(atomic_nums)
    return new_structure


class CatalystOrrDataset(Dataset):
    """Dataset that pairs catalyst structures with ORR overpotential labels."""

    def __init__(self, structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4],
                 use_binary_labels=True, normalize_target=False, top_n_high_performance=64,
                 top_n_low_pt_fraction=64, label_threshold=0.3):
        self.grid_size = grid_size
        self.use_binary_labels = use_binary_labels
        self.normalize_target = normalize_target
        self.top_n_high_performance = top_n_high_performance
        self.top_n_low_pt_fraction = top_n_low_pt_fraction
        self.label_threshold = label_threshold
        
        if isinstance(structures_db_paths, str):
            structures_db_paths = [structures_db_paths]
            
        if isinstance(overpotentials_json_paths, str):
            overpotentials_json_paths = [overpotentials_json_paths]
            
        self.structures = {}
        for db_path in structures_db_paths:
            print(f"Loading structure database: {db_path}")
            if not os.path.exists(db_path):
                print(f"Warning: file not found: {db_path}")
                continue
                
            try:
                db = connect(db_path)
                for row in db.select():
                    uid = row.unique_id
                    self.structures[uid] = row.toatoms()
            except Exception as e:
                print(f"Error: failed to read {db_path}: {e}")
                continue
        
        print(f"Loaded {len(self.structures)} structures in total")
        print(f"Sample structure IDs: {list(self.structures.keys())[:3]}")
        
        self.overpotentials = []
        for json_path in overpotentials_json_paths:
            print(f"Loading overpotential data: {json_path}")
            if not os.path.exists(json_path):
                print(f"Warning: file not found: {json_path}")
                continue
                
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                    self.overpotentials.extend(data)
            except Exception as e:
                print(f"Error: failed to parse {json_path}: {e}")
                continue
        
        print(f"Loaded {len(self.overpotentials)} overpotential entries")
        print(f"Sample unique IDs: {[entry.get('unique_id') for entry in self.overpotentials[:3]]}")
        
        self.valid_indices = []
        self.raw_overpotentials = []
        self.raw_alloy_formations = []
        self.targets = []
        self.source_info = {}
        
        for entry in self.overpotentials:
            uid = entry.get('unique_id')
            eta = entry.get('overpotential')
            alloy_formation = entry.get('E_alloy_formation')
            
            if uid in self.structures and eta is not None and alloy_formation is not None:
                if uid in self.source_info:
                    idx = self.valid_indices.index(uid)
                    self.raw_overpotentials[idx] = eta
                    self.raw_alloy_formations[idx] = alloy_formation
                    self.source_info[uid] = entry
                else:
                    self.valid_indices.append(uid)
                    self.raw_overpotentials.append(eta)
                    self.raw_alloy_formations.append(alloy_formation)
                    self.source_info[uid] = entry
        
        print(f"Matched entries: {len(self.valid_indices)}")
        
        # Compute overpotential statistics
        if self.raw_overpotentials:
            self.overpotential_median = np.median(self.raw_overpotentials)
            self.overpotential_mean = np.mean(self.raw_overpotentials)
            self.overpotential_std = np.std(self.raw_overpotentials)
            self.overpotential_min = min(self.raw_overpotentials)
            self.overpotential_max = max(self.raw_overpotentials)
            
            print("Overpotential statistics:")
            print(f"  Range: {self.overpotential_min:.3f} ~ {self.overpotential_max:.3f} V")
            print(f"  Mean: {self.overpotential_mean:.3f} V")
            print(f"  Median: {self.overpotential_median:.3f} V")
        else:
            print("Warning: no valid overpotential data found")
            raise ValueError("Dataset is empty. Check the input files.")
        
        # Compute alloy formation energy statistics
        if self.raw_alloy_formations:
            self.alloy_formation_median = np.median(self.raw_alloy_formations)
            self.alloy_formation_mean = np.mean(self.raw_alloy_formations)
            self.alloy_formation_std = np.std(self.raw_alloy_formations)
            self.alloy_formation_min = min(self.raw_alloy_formations)
            self.alloy_formation_max = max(self.raw_alloy_formations)
            
            print("Alloy formation energy statistics:")
            print(f"  Range: {self.alloy_formation_min:.3f} ~ {self.alloy_formation_max:.3f} eV")
            print(f"  Mean: {self.alloy_formation_mean:.3f} eV")
            print(f"  Median: {self.alloy_formation_median:.3f} eV")
        
        if self.use_binary_labels:
            overpotential_threshold = np.percentile(self.raw_overpotentials, self.label_threshold * 100)
            overpotential_labels = [1 if eta < overpotential_threshold else 0 for eta in self.raw_overpotentials]
            
            alloy_formation_threshold = np.percentile(self.raw_alloy_formations, self.label_threshold * 100)
            alloy_formation_labels = [1 if formation < alloy_formation_threshold else 0 for formation in self.raw_alloy_formations]

            self.targets = [[overpotential_labels[i], alloy_formation_labels[i]] for i in range(len(self.raw_overpotentials))]

            high_performance_count = sum(overpotential_labels)
            stable_alloy_count = sum(alloy_formation_labels)

            print(f"Binary label summary (threshold={self.label_threshold:.1%}):")
            print(f"  High-performance catalysts (overpotential < {overpotential_threshold:.3f} V): {high_performance_count}")
            print(f"  Low-performance catalysts (overpotential >= {overpotential_threshold:.3f} V): {len(overpotential_labels) - high_performance_count}")
            print(f"  Stable alloys (formation energy < {alloy_formation_threshold:.3f} eV): {stable_alloy_count}")
            print(f"  Unstable alloys (formation energy >= {alloy_formation_threshold:.3f} eV): {len(alloy_formation_labels) - stable_alloy_count}")
            
        else:
            self.targets = self.raw_overpotentials.copy()
            if normalize_target:
                if abs(self.overpotential_max - self.overpotential_min) < 1e-6:
                    self.overpotential_min = self.overpotential_min - 0.5
                    self.overpotential_max = self.overpotential_max + 0.5
                self.targets = [
                    (eta - self.overpotential_min) / (self.overpotential_max - self.overpotential_min)
                    for eta in self.targets
                ]
                print("Using normalised continuous labels")
            else:
                print("Using raw continuous overpotential labels")
        
        print(f"Valid samples: {len(self.valid_indices)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        uid = self.valid_indices[idx]
        structure = self.structures[uid]
        target = self.targets[idx]
        
        atoms_sorted = sort_atoms(structure, axes=("z", "y", "x"))
        structure_tensor = structure_to_tensor(atoms_sorted, self.grid_size)
        result = atomic_numbers_tensor_to_classes(structure_tensor)
        
        if self.use_binary_labels:
            target_tensor = torch.tensor(target, dtype=torch.float32)
        else:
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
        return result, target_tensor
    
    def get_target_range(self):
        """Return the range used for normalisation."""
        return self.overpotential_min, self.overpotential_max

    def get_overpotential_stats(self):
        """Return summary statistics of the overpotential values."""
        return {
            'median': self.overpotential_median,
            'mean': self.overpotential_mean,
            'std': self.overpotential_std,
            'min': self.overpotential_min,
            'max': self.overpotential_max
        }

    def get_composition(self, idx):
        """Return elemental fractions for a sample as a dictionary."""
        uid = self.valid_indices[idx]
        entry = self.source_info.get(uid, {})
        structure = self.structures.get(uid)
        return extract_composition(entry, structure=structure)

    def get_raw_overpotential(self, idx):
        """Return the raw overpotential value for the given index."""
        return self.raw_overpotentials[idx]

    def get_raw_alloy_formation(self, idx):
        """Return the raw alloy formation energy for the given index."""
        return self.raw_alloy_formations[idx]

    def denormalize_target(self, normalized_value):
        """Convert a normalised value back to the original scale (continuous labels only)."""
        if not self.use_binary_labels and self.normalize_target:
            return normalized_value * (self.overpotential_max - self.overpotential_min) + self.overpotential_min
        else:
            return normalized_value

def create_dataset_from_json(structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4],
                           use_binary_labels=True, normalize_target=False, top_n_high_performance=64):
    """Convenience wrapper to build the dataset from JSON inputs."""
    dataset = CatalystOrrDataset(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size,
        use_binary_labels=use_binary_labels,
        normalize_target=normalize_target,
        top_n_high_performance=top_n_high_performance
    )
    return dataset

def calc_alloy_formation_energy(
    bulk_atoms,
    bulk_energy,
    calculator="fairchem",
    per_atom=True,
    cache_dir=None,
):
    """
    Compute the alloy formation energy for a slab.

    Parameters
    ----------
    bulk_atoms : ase.Atoms
        Optimised slab structure.
    bulk_energy : float
        Total energy of the slab.
    calculator : str
        Name of the calculator backend (default: ``"fairchem"``).
    per_atom : bool
        If ``True`` return energy per atom, otherwise return the total value.
    cache_dir : str | pathlib.Path | None
        Directory used to store bulk-reference cache files. When omitted,
        ``src/orr_vae/data`` is used.

    Returns
    -------
    float
        Formation energy in eV or eV/atom.
    """
    from pathlib import Path
    from collections import Counter
    from ase.build import fcc111
    
    try:
        from surface.orr_overpotential_calculator.calc_orr_energy import optimize_bulk_structure
    except ImportError:  # pragma: no cover - compatibility fallback
        from orr_overpotential_calculator.calc_orr_energy import optimize_bulk_structure
    
    cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).parent / "data"
    cache_root.mkdir(parents=True, exist_ok=True)
    bulk_data_path = cache_root / f"{calculator}_bulk_data.json"

    if bulk_data_path.exists():
        print(f"Loading existing bulk reference cache: {bulk_data_path}")
        with open(bulk_data_path, 'r') as f:
            bulk_data = json.load(f)
    else:
        print(f"Bulk reference cache not found. Creating: {bulk_data_path}")
        bulk_data = {}

    symbols = bulk_atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    missing_elements = [element for element in unique_elements if element not in bulk_data]

    if missing_elements:
        print(f"Missing bulk references will be computed: {missing_elements}")

    for element in missing_elements:
        print(f"Optimising 4x4x4 bulk for {element} ...")

        lattice_const = elemental_a(element)
        print(f"Lattice constant for {element}: {lattice_const:.4f} Å")

        element_bulk = fcc111(
            symbol=element,
            size=[4, 4, 4],
            a=lattice_const,
            vacuum=None,
            periodic=True,
        )

        work_dir = Path(f"temp_bulk_{element}")
        work_dir.mkdir(exist_ok=True)

        try:
            optimized_bulk, element_bulk_energy = optimize_bulk_structure(
                element_bulk, str(work_dir), calculator=calculator
            )
            bulk_data[element] = {
                "n_atoms": len(element_bulk),
                "energy": float(element_bulk_energy)
            }
            print(f"{element}: {len(element_bulk)} atoms, energy = {element_bulk_energy:.6f} eV")
        except Exception as e:
            print(f"Warning: failed to optimise bulk for {element}: {e}")
            bulk_data[element] = {
                "n_atoms": len(element_bulk),
                "energy": 0.0
            }
        finally:
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir)

    if missing_elements:
        with open(bulk_data_path, 'w') as f:
            json.dump(bulk_data, f, indent=2)
        print(f"Stored bulk reference data to {bulk_data_path}")
    
    # Alloy formation energy: E_form = E_slab^alloy - Σ(N_M^alloy × E_bulk(M) / N_bulk(M))
    
    # Count atoms for each element in the alloy
    symbols = bulk_atoms.get_chemical_symbols()
    element_counts = Counter(symbols)
    
    print(f"Alloy composition: {dict(element_counts)}")
    
    reference_energy = 0.0
    for element, count in element_counts.items():
        if element in bulk_data:
            element_energy_per_atom = bulk_data[element]["energy"] / bulk_data[element]["n_atoms"]
            reference_energy += count * element_energy_per_atom
            print(f"{element}: {count} atoms × {element_energy_per_atom:.6f} eV/atom = {count * element_energy_per_atom:.6f} eV")
        else:
            print(f"Warning: no bulk reference found for {element}")

    formation_energy = bulk_energy - reference_energy
    
    print(f"Alloy energy: {bulk_energy:.6f} eV")
    print(f"Reference energy: {reference_energy:.6f} eV")
    print(f"Formation energy: {formation_energy:.6f} eV")

    if per_atom:
        total_atoms = len(bulk_atoms)
        formation_energy_per_atom = formation_energy / total_atoms
        print(f"Formation energy per atom: {formation_energy_per_atom:.6f} eV/atom")
        return formation_energy_per_atom
    else:
        return formation_energy


def make_data_loaders_from_json(structures_db_paths, overpotentials_json_paths,
                               train_ratio=0.9, batch_size=16,
                               num_workers=0, seed=0,
                               grid_size=[4, 4, 4],
                               use_binary_labels=True, normalize_target=False,
                               top_n_high_performance=64, top_n_stable_alloy=64,
                               label_threshold=0.3):
    """Create train/test dataloaders for the ORR dataset with dual conditioning labels."""

    print("Preparing data loaders ...")
    print(f"Structure DBs: {structures_db_paths}")
    print(f"Overpotential JSONs: {overpotentials_json_paths}")
    print(f"Grid size: {grid_size}")
    print(f"Binary labels enabled: {use_binary_labels}")
    print(f"Top high-performance samples: {top_n_high_performance}")
    print(f"Top stable alloys: {top_n_stable_alloy}")
    
    dataset = CatalystOrrDataset(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size,
        use_binary_labels=use_binary_labels,
        normalize_target=normalize_target,
        top_n_high_performance=top_n_high_performance,
        top_n_low_pt_fraction=top_n_stable_alloy,
        label_threshold=label_threshold
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check the input files.")
    
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    
    print(f"Split sizes -> train: {n_train}, test: {n_test}")

    if n_train == 0 or n_test == 0:
        print("Warning: either train or test split is empty. Adjust train_ratio.")
    
    g = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=g)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Created data loaders (batch_size={batch_size})")
    
    return train_loader, test_loader, dataset
