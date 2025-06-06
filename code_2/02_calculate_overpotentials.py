#!/usr/bin/env python3
"""
ORR過電圧計算（モック実装）
実際の計算は外部ライブラリorr_overpotential_calculatorを使用することを想定
"""
import os
import json
import argparse
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

def mock_calc_orr_overpotential(atoms):
    """
    モックORR過電圧計算関数
    実際の実装では外部ライブラリを使用
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        float: ORR overpotential (V)
    """
    # 簡単なモック計算（Pt含有量とランダム要素に基づく）
    numbers = atoms.get_atomic_numbers()
    pt_count = np.sum(numbers == atomic_numbers["Pt"])
    ni_count = np.sum(numbers == atomic_numbers["Ni"])
    total_atoms = len(numbers)
    
    pt_fraction = pt_count / total_atoms
    
    # モック計算: Pt含有量が高いほど過電圧が低くなる傾向
    # + ランダム要素を追加
    base_overpotential = 0.8 - 0.6 * pt_fraction  # 0.2-0.8V range
    noise = np.random.normal(0, 0.1)  # ノイズ
    overpotential = max(0.05, base_overpotential + noise)  # 最低0.05V
    
    return overpotential

def calculate_overpotentials_from_json(structures_file, output_file):
    """
    JSON構造ファイルからORR過電圧を計算
    
    Args:
        structures_file: 構造データのJSONファイル
        output_file: 過電圧結果のJSONファイル
    """
    # 構造データの読み込み
    with open(structures_file, 'r') as f:
        structures = json.load(f)
    
    results = []
    
    print(f"Calculating overpotentials for {len(structures)} structures...")
    
    for key, structure_data in structures.items():
        unique_id = structure_data["unique_id"]
        numbers = np.array(structure_data["numbers"])
        positions = np.array(structure_data["positions"])
        cell = np.array(structure_data["cell"])
        pbc = structure_data["pbc"]
        
        # ASE Atomsオブジェクトの再構築
        atoms = Atoms(numbers=numbers, 
                     positions=positions, 
                     cell=cell, 
                     pbc=pbc)
        
        # ORR過電圧計算
        overpotential = mock_calc_orr_overpotential(atoms)
        
        # 結果データの作成
        result_data = {
            "unique_id": unique_id,
            "overpotential": float(overpotential),
            "chemical_formula": structure_data["chemical_formula"],
            "ni_fraction": structure_data["ni_fraction"],
            "pt_fraction": structure_data["pt_fraction"]
        }
        
        results.append(result_data)
        
        if len(results) % 10 == 0:
            print(f"Calculated {len(results)}/{len(structures)} overpotentials")
    
    # 結果をJSONファイルに保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Overpotentials saved to {output_file}")
    
    # 統計情報の表示
    overpotentials = [r["overpotential"] for r in results]
    print(f"Overpotential statistics:")
    print(f"  Mean: {np.mean(overpotentials):.3f} V")
    print(f"  Std:  {np.std(overpotentials):.3f} V")
    print(f"  Min:  {np.min(overpotentials):.3f} V")
    print(f"  Max:  {np.max(overpotentials):.3f} V")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Calculate ORR overpotentials")
    parser.add_argument("--structures_file", type=str, required=True,
                        help="Input structures JSON file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output overpotentials JSON file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # ランダムシード設定
    np.random.seed(args.seed)
    
    calculate_overpotentials_from_json(args.structures_file, args.output_file)

if __name__ == "__main__":
    main()