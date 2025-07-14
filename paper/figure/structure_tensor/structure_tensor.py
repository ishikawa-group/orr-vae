import numpy as np
import torch
from ase import Atoms
from ase.build import fcc111
from ase.io import write
import random

def create_pt_ni_slab(size=(4, 4, 4), ni_fraction=0.5, a=3.92, vacuum=None):
    """
    Pt-Ni合金のfcc(111)スラブ構造を作成します。
    
    Parameters:
      size (tuple): スラブのサイズ (x, y, z)
      ni_fraction (float): Ni原子の割合（0.0-1.0）
      a (float): 格子定数（Ptの格子定数：3.92 Å）
      vacuum (float): 真空層の厚さ（Noneの場合は真空層なし）
      
    Returns:
      atoms (ase.Atoms): Pt-Ni合金スラブ構造
    """
    # fcc(111)面のスラブを作成
    slab = fcc111('Pt', size=size, a=a, vacuum=vacuum)
    
    # 原子数を取得
    n_atoms = len(slab)
    n_ni = int(n_atoms * ni_fraction)
    
    # ランダムにNi原子を置換
    ni_indices = random.sample(range(n_atoms), n_ni)
    
    # 化学記号を設定
    symbols = ['Pt'] * n_atoms
    for i in ni_indices:
        symbols[i] = 'Ni'
    
    slab.set_chemical_symbols(symbols)
    
    return slab

def save_structure(atoms, filename_base):
    """
    原子構造を.xyzと.pngの両方の形式で保存します。
    
    Parameters:
      atoms (ase.Atoms): 保存する原子構造
      filename_base (str): ファイル名のベース（拡張子なし）
    """
    import os
    
    # 現在のディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # XYZファイルとして保存
    xyz_path = os.path.join(current_dir, f"{filename_base}.xyz")
    atoms.write(xyz_path)
    
    # PNGファイルとして保存
    png_path = os.path.join(current_dir, f"{filename_base}.png")
    write(png_path, atoms, rotation='-70x')

def save_tensor_to_txt(tensor, filename):
    """
    テンソルをテキストファイルに保存します。
    各z層（チャンネル）を8x8行列として記載します。
    
    Parameters:
      tensor (torch.Tensor): 保存するテンソル (z, y, x)
      filename (str): 保存するファイル名
    """
    import os
    
    z_size, y_size, x_size = tensor.shape
    
    # 現在のディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(current_dir, filename)
    
    with open(txt_path, 'w') as f:
        f.write(f"# Tensor shape: {tensor.shape}\n")
        f.write(f"# z_size: {z_size}, y_size: {y_size}, x_size: {x_size}\n\n")
        
        for z in range(z_size):
            f.write(f"# Channel {z+1} (z={z})\n")
            for y in range(y_size):
                row = " ".join([f"{tensor[z, y, x].item():6.1f}" for x in range(x_size)])
                f.write(f"{row}\n")
            f.write("\n")

def sort_atoms(atoms, axes=("z", "y", "x")):
    """
    Atoms オブジェクトを指定された軸順（デフォルトは (z, y, x)）でソートします。
    
    Parameters:
      atoms (ase.Atoms): ソート対象の原子構造
      axes (tuple): ソートに用いる軸。例: ("z", "y", "x")
      
    Returns:
      sorted_atoms (ase.Atoms): 指定した軸順にソートされた Atoms オブジェクト
    """
    import numpy as np
    
    axis_map = {"x": 0, "y": 1, "z": 2}
    pos = atoms.get_positions()  # shape: (n_atoms, 3)
    
    # lexsort：最後に与えたキーが最優先となるので、axes[::-1] として渡す
    keys = tuple(pos[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)
    
    sorted_atoms = atoms[sorted_indices]
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())
    
    return sorted_atoms

def structure_to_tensor(structure, grid_size):
    """
    fcc(111) の三角格子 (ABCABC…) を 3D テンソルへエンコードする。

    手順
    ----
    1. 原子を z→y→x の順でソートし (z, y, x) 形状に reshape。
    2. x, y 方向を 2 倍に拡張し、(row, col) の偶奇で
       A 層 = (偶,偶)・B 層 = (奇,奇)・C 層 = (偶,奇) にマッピングする。

    Parameters
    ----------
    structure : ase.Atoms
        原子数が x*y*z と一致する fcc(111) スラブ（例: fcc111('Pt', size=[x,y,z])）
    grid_size : (int, int, int)
        元のセル数 [x, y, z]。z は 3 の倍数推奨（ABC が 1 周で揃う）

    Returns
    -------
    torch.Tensor
        shape = (z, 2*y, 2*x)  の整数テンソル。
        0 は空セル、原子の Z 番号が値として入る。
    """
    import torch
    x_size, y_size, z_size = grid_size
    if len(structure) != x_size * y_size * z_size:
        raise ValueError("原子数と grid_size が一致しません")

    # (z,y,x) に並べ替えて 3D 配列化
    sorted_atoms = sort_atoms(structure, axes=("z", "y", "x"))
    basic = torch.tensor(sorted_atoms.get_atomic_numbers(),
                         dtype=torch.int64).reshape(z_size, y_size, x_size)

    # 出力テンソルを初期化 (空セル=0)
    interleaved = torch.zeros((z_size, 2*y_size, 2*x_size), dtype=torch.int64)

    for z in range(z_size):
        layer = basic[z]
        mod = z % 3            # 0:A, 1:B, 2:C
        if mod == 0:           # ----- A 層  (偶, 偶)
            interleaved[z, 0::2, 0::2] = layer
        elif mod == 1:         # ----- B 層  (奇, 奇)
            interleaved[z, 1::2, 1::2] = layer
        else:                  # ----- C 層  (偶, 奇)
            interleaved[z, 0::2, 1::2] = layer

    return interleaved


def main():
    """
    メイン関数：Pt-Ni合金スラブの作成、保存、テンソル変換を実行
    """
    # ランダムシードを設定（再現性のため）
    random.seed(42)
    np.random.seed(42)
    
    # Pt-Ni合金スラブを作成
    print("Creating Pt-Ni alloy slab...")
    slab = create_pt_ni_slab(size=(4, 4, 4), ni_fraction=0.5, vacuum=None)
    
    print(f"Created slab with {len(slab)} atoms")
    print(f"Chemical symbols: {slab.get_chemical_symbols()}")
    
    # 構造を保存
    print("Saving structure...")
    save_structure(slab, "pt_ni_slab_4x4x4")
    
    # テンソルに変換
    print("Converting to tensor...")
    tensor = structure_to_tensor(slab, grid_size=[4, 4, 4])
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    # テンソルを保存
    print("Saving tensor...")
    save_tensor_to_txt(tensor, "pt_ni_tensor_8x8x4.txt")
    
    print("All operations completed successfully!")
    
    # テンソルの内容を確認
    print("\nTensor content preview:")
    for z in range(tensor.shape[0]):
        print(f"Channel {z+1} (z={z}):")
        print(tensor[z])
        print()

if __name__ == "__main__":
    main()