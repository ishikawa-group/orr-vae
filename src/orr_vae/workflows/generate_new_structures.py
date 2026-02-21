#!/usr/bin/env python
"""
Generate new catalyst structures using a trained conditional VAE.
"""
import os
import argparse
import torch
import numpy as np
import importlib.util
import sys
from pathlib import Path
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
from orr_vae.tool import (
    ALLOY_ELEMENTS,
    NUM_CLASSES,
    class_tensor_to_atomic_numbers,
    sort_atoms,
    tensor_to_structure,
    vegard_lattice_constant,
)

import torch.nn.functional as F

def load_vae_class():
    """Dynamic import helper for ``ConditionalVAE`` defined in 03_conditional_vae.py."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vae_script_path = os.path.join(current_dir, "03_conditional_vae.py")
    
    if not os.path.exists(vae_script_path):
        raise FileNotFoundError(f"VAE script not found: {vae_script_path}")
    
    print(f"Loading VAE script: {vae_script_path}")
    
    original_argv = sys.argv.copy()
    sys.argv = ['03_conditional_vae.py']
    
    try:
        spec = importlib.util.spec_from_file_location("conditional_vae", vae_script_path)
        conditional_vae_module = importlib.util.module_from_spec(spec)
        sys.modules["conditional_vae"] = conditional_vae_module
        spec.loader.exec_module(conditional_vae_module)
        return conditional_vae_module.ConditionalVAE
    finally:
        sys.argv = original_argv

def convert_tensor_to_atomic_numbers(tensor, n_layers):
    """
    Convert generated logits into atomic-number layers.
    Tensor channels are arranged as NUM_CLASSES classes per layer.
    """
    expected_channels = n_layers * NUM_CLASSES
    if tensor.shape[0] != expected_channels:
        raise ValueError(
            f"Decoder output channel mismatch: got {tensor.shape[0]}, expected {expected_channels}"
        )

    _, height, width = tensor.shape
    discrete_tensor = torch.zeros(n_layers, height, width, dtype=torch.long)

    for layer in range(n_layers):
        start = layer * NUM_CLASSES
        end = start + NUM_CLASSES
        layer_logits = tensor[start:end]
        layer_probs = F.softmax(layer_logits, dim=0)
        layer_discrete = torch.argmax(layer_probs, dim=0)
        discrete_tensor[layer] = layer_discrete

    return class_tensor_to_atomic_numbers(discrete_tensor)

def calculate_composition(atomic_numbers_tensor):
    """
    Compute composition for all configured alloy elements from the atomic-number tensor.
    """
    flat_tensor = atomic_numbers_tensor.flatten()
    total_atoms = torch.count_nonzero(flat_tensor).item()

    element_counts = {}
    for element in ALLOY_ELEMENTS:
        z_num = atomic_numbers[element]
        count = int(torch.sum(flat_tensor == z_num).item())
        element_counts[element] = count

    if total_atoms == 0:
        return {element: 0.0 for element in ALLOY_ELEMENTS}, element_counts

    fractions = {element: count / total_atoms for element, count in element_counts.items()}
    return fractions, element_counts

def create_template_structure(composition, size, vacuum):
    """
    Build a template structure using Vegard's law and sort atoms in a canonical order.
    """
    fractions = [composition.get(element, 0.0) for element in ALLOY_ELEMENTS]
    total = sum(fractions)
    if total <= 0:
        fractions = [1.0 / len(ALLOY_ELEMENTS)] * len(ALLOY_ELEMENTS)
    else:
        fractions = [fraction / total for fraction in fractions]
    
    lattice_const = vegard_lattice_constant(ALLOY_ELEMENTS, fractions)
    
    bulk = fcc111(symbol=ALLOY_ELEMENTS[0], 
                  size=size, 
                  a=lattice_const,
                  vacuum=vacuum, 
                  periodic=True)
    
    bulk_sorted = sort_atoms(bulk, axes=("z", "y", "x"))
    
    return bulk_sorted, lattice_const

def check_atoms_numbers_duplicate(structure1, structure2):
    """
    Check for duplicates by comparing ``atoms.numbers`` arrays.
    """
    try:
        numbers1 = structure1.get_atomic_numbers()
        numbers2 = structure2.get_atomic_numbers()
        
        if len(numbers1) != len(numbers2):
            return False
        
        return np.array_equal(numbers1, numbers2)
    except Exception as e:
        print(f"Warning: duplicate check failed: {e}")
        return False

def load_existing_structures(data_dir, iter_names):
    """
    Load existing structures for duplicate checking.
    Returns both the ``Atoms`` objects and their atomic numbers for quick comparisons.
    """
    existing_structures = []
    existing_numbers = []
    
    for iter_name in iter_names:
        db_path = os.path.join(data_dir, f"{iter_name}_structures.json")
        if os.path.exists(db_path):
            print(f"Loading existing structures: {db_path}")
            try:
                db = connect(db_path)
                count = 0
                for row in db.select():
                    atoms = row.toatoms(add_additional_information=True)
                    d = atoms.info.pop("data", {})
                    atoms.info["adsorbate_info"] = d.get("adsorbate_info", {})
                    
                    existing_structures.append(atoms)
                    existing_numbers.append(atoms.get_atomic_numbers())
                    count += 1
                print(f"  Loaded {count} structures")
            except Exception as e:
                print(f"Warning: failed to read {db_path}: {e}")
        else:
            print(f"Warning: {db_path} not found")
    
    print(f"Total existing structures loaded: {len(existing_structures)}")
    return existing_structures, existing_numbers

def generate_structures():
    """
    Generate new alloy structures using the trained conditional VAE.
    """
    parser = argparse.ArgumentParser(description="Generate fcc(111) alloy slabs using a trained conditional VAE")
    parser.add_argument("--iter", type=int, default=3,
                        help="Iteration index (default: 3)")
    parser.add_argument("--num", type=int, default=128,
                        help="Number of structures to generate (default: 128)")
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent / "data"),
                        help="Output directory for generated structures")
    parser.add_argument("--result_dir", type=str,
                        default=str(Path(__file__).parent / "result"),
                        help="Directory containing trained VAE checkpoints")
    parser.add_argument("--vae_model_path", type=str, default=None,
                        help="Optional path to a trained VAE checkpoint (auto-resolved when omitted)")
    parser.add_argument("--overpotential_condition", type=int, choices=[0, 1], default=1,
                        help="Overpotential condition (0: high, 1: low; default: 1)")
    parser.add_argument("--alloy_stability_condition", type=int, choices=[0, 1], default=1,
                        help="Alloy stability condition (0: unstable, 1: stable; default: 1)")
    parser.add_argument("--latent_size", type=int, default=32,
                        help="Latent dimensionality (default: 32)")
    parser.add_argument("--existing_iters", type=str, nargs='+', default=None,
                        help="List of iteration names used for duplicate checking (auto-detected when omitted)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--grid_x", type=int, default=4,
                        help="Grid size along x (default: 4)")
    parser.add_argument("--grid_y", type=int, default=4,
                        help="Grid size along y (default: 4)")
    parser.add_argument("--grid_z", type=int, default=4,
                        help="Grid size along z / number of slab layers (default: 4)")
    args = parser.parse_args()

    ITER = args.iter
    
    if args.vae_model_path is None:
        args.vae_model_path = os.path.join(args.result_dir, f"iter{ITER}", f"final_cvae_iter{ITER}.pt")
    
    if args.existing_iters is None:
        args.existing_iters = [f"iter{i}" for i in range(ITER + 1)]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    size = [args.grid_x, args.grid_y, args.grid_z]
    vacuum = None
    
    print("=== Structure generation using trained VAE ===")
    print(f"Current iteration: {ITER}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {args.output_dir}")
    print(f"Result directory: {args.result_dir}")
    print(f"VAE checkpoint: {args.vae_model_path}")
    print(f"Overpotential condition: {args.overpotential_condition} ({'low' if args.overpotential_condition == 1 else 'high'})")
    print(f"Alloy stability condition: {args.alloy_stability_condition} ({'stable' if args.alloy_stability_condition == 1 else 'unstable'})")
    print(f"Number of structures to generate: {args.num}")
    print(f"Grid size: {size}")
    print(f"Latent size: {args.latent_size}")
    print(f"Duplicate check iterations: {args.existing_iters}")
    print("Duplicate checking method: exact match on atoms.numbers")
    
    ConditionalVAE = load_vae_class()
    
    vae_model = ConditionalVAE(
        latent_size=args.latent_size,
        condition_dim=2,
        structure_layers=args.grid_z,
    ).to(DEVICE)
    
    if not os.path.exists(args.vae_model_path):
        print(f"Error: VAE checkpoint not found: {args.vae_model_path}")
        return
    
    vae_model.load_state_dict(torch.load(args.vae_model_path, map_location=DEVICE))
    vae_model.eval()
    
    print("Loaded VAE weights.")
    
    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    existing_structures, existing_numbers = load_existing_structures(data_dir, args.existing_iters)

    next_iter = ITER + 1
    db_path = os.path.join(data_dir, f"iter{next_iter}_structures.json")
    db = connect(db_path)
    
    print(f"Target output: iter{next_iter}_structures.json")
    print(f"Generating {args.num} unique structures ...")
    print(f"Duplicate check pool: {len(existing_structures)} existing structures")
    
    unique_structures = existing_structures.copy()
    unique_numbers = existing_numbers.copy()
    successful_generations = 0
    max_attempts = args.num * 100000
    attempt = 0
    duplicate_with_existing = 0
    duplicate_with_new = 0
    
    with torch.no_grad():
        while successful_generations < args.num and attempt < max_attempts:
            try:
                attempt += 1
                
                z = torch.randn(1, args.latent_size).to(DEVICE)
                condition = torch.tensor([[args.overpotential_condition, args.alloy_stability_condition]],
                                       dtype=torch.float32).to(DEVICE)  # [1, 2]
                
                generated_tensor = vae_model.decode(z, condition)
                
                if generated_tensor.dim() == 4:
                    generated_tensor = generated_tensor.squeeze(0)
                
                atomic_numbers_tensor = convert_tensor_to_atomic_numbers(
                    generated_tensor,
                    args.grid_z,
                )
                
                composition, element_counts = calculate_composition(atomic_numbers_tensor)
                
                if sum(element_counts.values()) == 0:
                    continue

                template_structure, lattice_const = create_template_structure(
                    composition, size, vacuum)
                
                final_structure = tensor_to_structure(atomic_numbers_tensor, template_structure)

                chemical_formula = final_structure.get_chemical_formula()
                if 'X' in chemical_formula:
                    if attempt % 100 == 0:
                        print(f"Attempt {attempt}: skipped structure containing placeholder atom X "
                              f"(success: {successful_generations}/{args.num})")
                    continue

                current_numbers = final_structure.get_atomic_numbers()
                is_duplicate = False
                
                for i, existing_nums in enumerate(unique_numbers):
                    if np.array_equal(current_numbers, existing_nums):
                        is_duplicate = True
                        if i < len(existing_numbers):
                            duplicate_with_existing += 1
                        else:
                            duplicate_with_new += 1
                        break
                
                if is_duplicate:
                    if attempt % 100 == 0:
                        print(
                            f"Attempt {attempt}: skipped duplicate structure "
                            f"(success: {successful_generations}/{args.num}, "
                            f"existing duplicates: {duplicate_with_existing}, "
                            f"new duplicates: {duplicate_with_new})"
                        )
                    continue
                
                unique_structures.append(final_structure.copy())
                unique_numbers.append(current_numbers.copy())
                
                final_structure.calc = EMT()

                ads_info = final_structure.info.get("adsorbate_info", {})
                
                data = {
                    "chemical_formula": final_structure.get_chemical_formula(),
                    "composition": {element: float(composition[element]) for element in ALLOY_ELEMENTS},
                    "element_counts": element_counts,
                    "lattice_constant": float(lattice_const),
                    "run": successful_generations + 1,
                    "generation_method": "conditional_vae_dual_labels",
                    "overpotential_condition": int(args.overpotential_condition),
                    "alloy_stability_condition": int(args.alloy_stability_condition),
                    "adsorbate_info": ads_info,
                    "total_attempts": attempt,
                }
                for element in ALLOY_ELEMENTS:
                    data[f"{element.lower()}_fraction"] = float(composition[element])
                # Keep legacy keys for downstream compatibility.
                data["ni_fraction"] = float(composition.get("Ni", 0.0))
                data["pt_fraction"] = float(composition.get("Pt", 0.0))
                
                db.write(final_structure, data=data)
                successful_generations += 1
                
                print(f"Generated {successful_generations}/{args.num} structures (attempts: {attempt})")
                    
            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                continue
    
    print("\n=== Generation summary ===")
    print(f"Successful generations: {successful_generations}/{args.num}")
    print(f"Total attempts: {attempt}")
    if attempt:
        print(f"Success rate: {successful_generations / attempt * 100:.2f}%")
    print(f"Duplicates against existing structures: {duplicate_with_existing}")
    print(f"Duplicates among new structures: {duplicate_with_new}")
    print(f"Total duplicates skipped: {duplicate_with_existing + duplicate_with_new}")
    print(f"Structures saved to: {db_path}")
    print("Duplicate check: exact match of atoms.numbers arrays")
    
    if successful_generations < args.num:
        print("Warning: target count was not reached. Consider adjusting parameters.")
    
    print("Generation finished.")

def main():
    generate_structures()

if __name__ == "__main__":
    main()
