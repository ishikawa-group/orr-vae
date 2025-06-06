#!/usr/bin/env python3
"""
iter0用: Pt4x4x4構造からランダムに1-100%をNiで置換した構造を生成
"""
import os
import argparse
import json
import uuid
import numpy as np
from ase.build import fcc100
from ase.calculators.emt import EMT
from ase.data import atomic_numbers
from tool import vegard_lattice_constant, convert_numpy_types

def create_random_alloy_structures(num_structures=100, output_dir="./data"):
    """
    Pt4x4x4構造からランダムにNiで置換した構造を生成
    
    Args:
        num_structures: 生成する構造数
        output_dir: 出力ディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # パラメータ設定
    size = [4, 4, 4]  # 4x4x4構造
    vacuum = None
    alloy_elements = ["Pt", "Ni"]
    
    structures = {}
    
    print(f"Generating {num_structures} random alloy structures...")
    
    for i in range(num_structures):
        # Ni置換率をランダムに決定（1-100%）
        ni_fraction = np.random.uniform(0.01, 1.0)
        pt_fraction = 1.0 - ni_fraction
        fractions = [pt_fraction, ni_fraction]
        
        # Vegard法で格子定数を計算
        lattice_const = vegard_lattice_constant(alloy_elements, fractions)
        
        # fcc100表面の作成（基本はPtの構造）
        surf = fcc100(symbol="Pt", 
                      size=size, 
                      a=lattice_const,
                      vacuum=vacuum, 
                      periodic=True)
        
        # 合金組成の配置
        natoms = len(surf)  # 総原子数 (4*4*4=64)
        n_ni = int(round(natoms * ni_fraction))
        n_pt = natoms - n_ni
        
        # 原子番号リストを作成
        alloy_list = ([atomic_numbers["Pt"]] * n_pt + 
                     [atomic_numbers["Ni"]] * n_ni)
        
        # ランダムにシャッフル
        np.random.shuffle(alloy_list)
        surf.set_atomic_numbers(alloy_list)
        
        # EMT計算器の設定
        surf.calc = EMT()
        
        # ユニークIDの生成
        unique_id = str(uuid.uuid4()).replace('-', '')
        
        # 構造データの作成
        structure_data = {
            "unique_id": unique_id,
            "numbers": surf.get_atomic_numbers().tolist(),
            "positions": surf.get_positions().tolist(),
            "cell": surf.get_cell().array.tolist(),
            "pbc": surf.get_pbc().tolist(),
            "chemical_formula": surf.get_chemical_formula(),
            "ni_fraction": ni_fraction,
            "pt_fraction": pt_fraction,
            "lattice_constant": lattice_const,
            "run": i
        }
        
        # NumPy型を標準Python型に変換
        structure_data = convert_numpy_types(structure_data)
        
        structures[str(i+1)] = structure_data
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_structures} structures")
    
    # JSONファイルに保存
    output_file = os.path.join(output_dir, "iter0_structures.json")
    with open(output_file, 'w') as f:
        json.dump(structures, f, indent=2)
    
    print(f"Structures saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate random Pt-Ni alloy structures")
    parser.add_argument("--num", type=int, default=100,
                        help="Number of structures to generate (default: 100)")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory (default: ./data)")
    
    args = parser.parse_args()
    
    # ランダムシード設定（再現性のため）
    np.random.seed(42)
    
    create_random_alloy_structures(args.num, args.output_dir)

if __name__ == "__main__":
    main()