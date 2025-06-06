#!/usr/bin/env python3
"""
学習済みGANから新しい触媒構造を生成
"""
import os
import json
import argparse
import uuid
import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from ase.build import fcc100
from ase.data import atomic_numbers
from tool import tensor_to_slab, vegard_lattice_constant, convert_numpy_types

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    """生成器（03_conditional_gan.pyと同じ構造）"""
    
    def __init__(self, noise_dim=128, condition_dim=2, channels=4):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        # 条件ラベルの埋め込み
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        
        # ノイズ+条件の線形変換
        self.linear = nn.Sequential(
            nn.Linear(noise_dim + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64 * 2 * 2),
            nn.BatchNorm1d(64 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # 転置畳み込み層
        self.deconv_layers = nn.Sequential(
            # 64x2x2 -> 32x4x4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x4x4 -> 16x8x8
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 16x8x8 -> 4x8x8
            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)  # 各ピクセルで確率分布
        )
    
    def forward(self, noise, conditions):
        # 条件の埋め込み
        condition_embed = self.condition_embed(conditions)
        
        # ノイズと条件を結合
        x = torch.cat([noise, condition_embed], dim=1)
        
        # 線形変換
        x = self.linear(x)
        x = x.view(x.size(0), 64, 2, 2)
        
        # 転置畳み込み
        x = self.deconv_layers(x)
        
        return x

def tensor_to_atoms(tensor, base_atoms=None):
    """
    4チャンネルテンソルをASE Atomsオブジェクトに変換
    
    Args:
        tensor: 4x8x8テンソル（確率分布）
        base_atoms: ベース構造（Pt4x4x4）
        
    Returns:
        ASE Atoms object
    """
    if base_atoms is None:
        # デフォルトのPt4x4x4構造を作成
        base_atoms = fcc100(symbol="Pt", size=[4, 4, 4], a=3.92, periodic=True)
    
    # 確率分布から最も可能性の高い元素を選択
    # 0: 空サイト（無視）, 1: Ni, 2: Pt
    predicted_elements = torch.argmax(tensor, dim=0)  # 8x8
    
    # 4x4x4グリッドに戻す
    # tensor_to_slabを使用するために、まず適切な形状に変換
    reconstructed_tensor = torch.zeros(4, 4, 4, dtype=torch.float64)
    
    # 8x8から4x4x4への逆変換（slab_to_tensorの逆操作）
    for z in range(4):
        if z % 2 == 0:
            # 偶数層：[0::2, 0::2]から抽出
            layer = predicted_elements[0::2, 0::2] if z == 0 else predicted_elements[0::2, 0::2]
        else:
            # 奇数層：[1::2, 1::2]から抽出
            layer = predicted_elements[1::2, 1::2]
        
        # 元素マッピング: 1->Ni(28), 2->Pt(78), 0->Pt(デフォルト)
        layer_mapped = torch.where(layer == 1, 
                                  torch.tensor(atomic_numbers["Ni"]),
                                  torch.tensor(atomic_numbers["Pt"]))
        reconstructed_tensor[z] = layer_mapped.float()
    
    # tensor_to_slabを使用してASE Atomsに変換
    try:
        atoms = tensor_to_slab(reconstructed_tensor, base_atoms)
    except Exception as e:
        print(f"Error in tensor_to_slab: {e}")
        # フォールバック：手動で原子番号を設定
        atoms = base_atoms.copy()
        flat_numbers = reconstructed_tensor.flatten().numpy().astype(int)
        atoms.set_atomic_numbers(flat_numbers)
    
    return atoms

def generate_new_structures(model_path, num_structures=100, output_dir="./data", iter_num=1):
    """
    学習済みGANから新しい構造を生成
    
    Args:
        model_path: 学習済み生成器のパス
        num_structures: 生成する構造数
        output_dir: 出力ディレクトリ
        iter_num: イテレーション番号
    """
    # モデルの読み込み
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # ベース構造（Pt4x4x4）の作成
    base_atoms = fcc100(symbol="Pt", size=[4, 4, 4], a=3.92, periodic=True)
    
    structures = {}
    
    print(f"Generating {num_structures} new structures with target conditions...")
    
    with torch.no_grad():
        for i in range(num_structures):
            # ランダムノイズの生成
            noise = torch.randn(1, 128, device=device)
            
            # 目標条件：ORR過電圧ラベル1, Pt含有量ラベル1
            target_conditions = torch.ones(1, 2, device=device)
            
            # 構造生成
            generated_tensor = generator(noise, target_conditions)
            generated_tensor = generated_tensor.squeeze(0).cpu()  # 4x8x8
            
            # ASE Atomsオブジェクトに変換
            try:
                atoms = tensor_to_atoms(generated_tensor, base_atoms)
                
                # 合金組成の計算
                numbers = atoms.get_atomic_numbers()
                ni_count = np.sum(numbers == atomic_numbers["Ni"])
                pt_count = np.sum(numbers == atomic_numbers["Pt"])
                total_atoms = len(numbers)
                
                ni_fraction = ni_count / total_atoms
                pt_fraction = pt_count / total_atoms
                
                # Vegard法で格子定数を計算
                if ni_fraction > 0:
                    fractions = [pt_fraction, ni_fraction]
                    lattice_const = vegard_lattice_constant(["Pt", "Ni"], fractions)
                    # 格子定数を更新
                    new_cell = atoms.get_cell().array * (lattice_const / 3.92)
                    atoms.set_cell(new_cell, scale_atoms=True)
                
                # ユニークIDの生成
                unique_id = str(uuid.uuid4()).replace('-', '')
                
                # 構造データの作成
                structure_data = {
                    "unique_id": unique_id,
                    "numbers": atoms.get_atomic_numbers().tolist(),
                    "positions": atoms.get_positions().tolist(),
                    "cell": atoms.get_cell().array.tolist(),
                    "pbc": atoms.get_pbc().tolist(),
                    "chemical_formula": atoms.get_chemical_formula(),
                    "ni_fraction": float(ni_fraction),
                    "pt_fraction": float(pt_fraction),
                    "lattice_constant": float(lattice_const) if ni_fraction > 0 else 3.92,
                    "run": i,
                    "generation_method": "conditional_gan",
                    "target_conditions": [1.0, 1.0]  # [eta_label, pt_label]
                }
                
                # NumPy型を標準Python型に変換
                structure_data = convert_numpy_types(structure_data)
                
                structures[str(i+1)] = structure_data
                
                if (i+1) % 10 == 0:
                    print(f"Generated {i+1}/{num_structures} structures")
                    
            except Exception as e:
                print(f"Error generating structure {i+1}: {e}")
                continue
    
    # JSONファイルに保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"iter{iter_num}_structures.json")
    with open(output_file, 'w') as f:
        json.dump(structures, f, indent=2)
    
    print(f"Generated structures saved to {output_file}")
    
    # 統計情報の表示
    if structures:
        ni_fractions = [s["ni_fraction"] for s in structures.values()]
        pt_fractions = [s["pt_fraction"] for s in structures.values()]
        
        print(f"Generation statistics:")
        print(f"  Ni fraction - Mean: {np.mean(ni_fractions):.3f}, Std: {np.std(ni_fractions):.3f}")
        print(f"  Pt fraction - Mean: {np.mean(pt_fractions):.3f}, Std: {np.std(pt_fractions):.3f}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate new structures using trained GAN")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained generator model")
    parser.add_argument("--num_structures", type=int, default=100,
                        help="Number of structures to generate")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory")
    parser.add_argument("--iter_num", type=int, default=1,
                        help="Iteration number")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # ランダムシード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    generate_new_structures(
        args.model_path, 
        args.num_structures, 
        args.output_dir, 
        args.iter_num
    )

if __name__ == "__main__":
    main()