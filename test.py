#!/usr/bin/env python
import torch
import numpy as np
from ase.build import fcc100
from ase.data import atomic_numbers  # 原子番号の辞書
from ase import Atoms

# --- 改良版 sort_atoms_by 関数 ---
def sort_atoms_by(atoms, axes=("z", "y", "x")):
    """
    Atoms オブジェクトを指定した軸の順（デフォルトは z, y, x）で
    レキシコグラフィカル（辞書順）にソートして新たな Atoms オブジェクトとして返す。
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    # 座標を取得 (n_atoms, 3)
    pos = atoms.get_positions()
    # 指定された軸順に従ってソートするため、最後のキーが最優先となるように逆順に渡す
    keys = tuple(pos[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)
    sorted_atoms = atoms[sorted_indices]
    # 元のタグ、セル、周期境界条件を復元
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())
    return sorted_atoms

# --- パラメータ設定 ---
# グリッドサイズ：x=8, y=8, z=3（＝3層のスラブ）
size = [8, 8, 3]  # [x, y, z]
vacuum = 15.0
lattice_const = 3.9   # Pt を基準とする格子定数
alloy_elements = ["Pt", "Rh", "Ir", "Pd"]

# fcc100 表面の作成（基本は Pt 構造）
surf = fcc100(symbol="Pt", size=size, a=lattice_const,
              vacuum=vacuum, orthogonal=True, periodic=True)

# --- 合金組成の均等配置 ---
natoms = len(surf)  # 表面内の原子数（例：192個）
base_count = natoms // len(alloy_elements)  # 各元素の基本個数
leftover = natoms % len(alloy_elements)      # 割り切れない余り分

# 各元素の原子番号を均等に配置するリストを作成
alloy_list = []
for element in alloy_elements:
    alloy_list.extend([atomic_numbers[element]] * base_count)
for j in range(leftover):
    alloy_list.append(atomic_numbers[alloy_elements[j]])

# ランダムにシャッフルして各元素をランダム配置
np.random.shuffle(alloy_list)
surf.set_atomic_numbers(alloy_list)

print(f"生成された表面の原子数: {len(surf)}")

# --- ソート：sort_atoms_by 関数を利用して (z, y, x) の順に整列 ---
surf_sorted = sort_atoms_by(surf, axes=("z", "y", "x"))

# --- Option 1: 単純な reshape によるテンソル化 ---
atomic_numbers_arr = surf_sorted.get_atomic_numbers()
try:
    # sorted な原子番号配列を (z, y, x) のグリッド（＝(3, 8, 8)）に reshape
    tensor_grid = torch.tensor(atomic_numbers_arr, dtype=torch.int64).reshape(size[2], size[1], size[0])
    print("Option 1: ソート後に直接 reshape")
except ValueError as e:
    print(f"Option 1 のエラー: {e}")
    tensor_grid = None

# --- Option 2: 原子位置に基づく整列によるテンソル化 ---
# scaled positions を使えば、原子のセル内での相対位置 (0～1) が取得できる
scaled_positions = surf_sorted.get_scaled_positions()
discretized_x = np.floor(scaled_positions[:, 0] * size[0]).astype(int)
discretized_y = np.floor(scaled_positions[:, 1] * size[1]).astype(int)
discretized_z = np.floor(scaled_positions[:, 2] * size[2]).astype(int)

# 3D 配列を初期化 (初期値は 0)
tensor_3d = np.zeros((size[2], size[1], size[0]), dtype=int)
for i, (x, y, z) in enumerate(zip(discretized_x, discretized_y, discretized_z)):
    if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
        tensor_3d[z, y, x] = atomic_numbers_arr[i]

tensor_grid_sorted = torch.tensor(tensor_3d, dtype=torch.int64)

# --- 結果の確認 ---
if tensor_grid is not None:
    print("\nOption 1 のテンソル（直接 reshape）:")
    print(tensor_grid.shape)
    print(tensor_grid)

print("\nOption 2 のテンソル（原子位置に基づく整列）:")
print(tensor_grid_sorted.shape)
print(tensor_grid_sorted)

expected_shape = (size[2], size[1], size[0])
print(f"\n期待される形状: {expected_shape}, Option 2 の実際の形状: {tensor_grid_sorted.shape}")

# 重複（同じグリッドセルに複数の原子が配置された位置）と空セルのチェック
unique_positions = set()
duplicates = []
for i, (x, y, z) in enumerate(zip(discretized_x, discretized_y, discretized_z)):
    pos = (x, y, z)
    if pos in unique_positions:
        duplicates.append(pos)
    unique_positions.add(pos)

if duplicates:
    print(f"\n警告: {len(duplicates)} 個の重複位置が見つかりました: {duplicates[:5]}...")
else:
    print("\n重複なし: 各位置に正確に 1 つの原子が配置されています")

zero_count = np.sum(tensor_3d == 0)
if zero_count > 0:
    print(f"\n警告: {zero_count} 個の空の位置があります")
else:
    print("\nすべての位置に原子が配置されています")

print("\n変換完了！")
