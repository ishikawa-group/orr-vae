def convert_numpy_types(obj):
    """NumPy型を標準Python型に変換する"""
    import numpy as np
    if isinstance(obj, np.number):
        return obj.item()  # NumPy数値型をPython標準の数値型に変換
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def elemental_a(symbol: str) -> float:
    """ASE の reference_states から FCC 格子定数 (Å) を返す"""
    from ase.data import reference_states, atomic_numbers
    Z = atomic_numbers[symbol]
    a = reference_states[Z].get('a')  # FCC は 'a' キーに格子定数
    if a is None:
        raise ValueError(f"No reference lattice constant for {symbol}")
    return a

def vegard_lattice_constant(elements, fractions=None):
    """
    elements : ['Pt','Ni', ...]
    fractions: [0.5,0.5] など。None の場合は等分
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
    原子の z 座標に基づいて、モデル内の層の数を計算する関数。
    
    Args:
        atoms: ASE atoms オブジェクト
        
    Returns:
        層の数（整数）
    """
    import numpy as np

    pos  = atoms.positions
    # 層の識別のための丸め（ここでは3桁で丸め）
    zpos = np.round(pos[:,2], decimals=3)
    nlayer = len(set(zpos))
    return nlayer


def set_tags_by_z(atoms):
    """
    原子の z 座標に基づいて、層ごとにタグを設定する関数。
    各層の原子には、下から順に 0, 1, 2... のタグが付けられる。
    
    Args:
        atoms: ASE atoms オブジェクト
        
    Returns:
        タグが設定された新しい atoms オブジェクト
    """
    import numpy as np
    import pandas as pd

    newatoms = atoms.copy()
    pos  = newatoms.positions
    # 小数第1位で丸め（層の幅の目安として利用）
    zpos = np.round(pos[:,2], decimals=1)
    
    # 一意な層の値を抽出し、必ず昇順にソートする
    bins = np.sort(np.array(list(set(zpos)))) + 1.0e-2
    bins = np.insert(bins, 0, 0)
    
    # 各区間にラベルを設定（0,1,2,...）
    labels = list(range(len(bins)-1))
    tags = pd.cut(zpos, bins=bins, labels=labels, include_lowest=True).tolist()
    newatoms.set_tags(tags)
    
    return newatoms

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


def structure_to_tensor(slab, grid_size):
    """
    結晶構造を、グリッドサイズ [x, y, z] に基づいて 3 次元テンソルに変換します。
    
    ※まず、ソート（axes=("z", "y", "x")）により (z, y, x) の形状に reshape し、
      その後、x・y 方向に交互に 0 を挿入します。
    ※さらに、各 z 層で、偶数層（z index even）の場合は基本パターン（even行・even列）、
      奇数層（z index odd）の場合はシフトしたパターン（odd行・odd列）に配置します。
    
    Parameters:
      slab (ase.Atoms): 変換対象の結晶構造（原子数は x*y*z と一致）
      grid_size (list or tuple): [x, y, z]（例: [8, 8, 3]）
      
    Returns:
      tensor (torch.Tensor): interleaved な 3 次元テンソル
        最終の shape は (z, new_y, new_x) で new_y = 2*y, new_x = 2*x
    """
    import torch
    import numpy as np
    
    x_size, y_size, z_size = grid_size  # grid_size = [x, y, z]
    total_cells = x_size * y_size * z_size
    
    if len(slab) != total_cells:
        raise ValueError(f"結晶内の原子数 {len(slab)} がグリッドセル数 {total_cells} と一致しません")
    
    # ソートして (z, y, x) の形状に reshape
    sorted_slab = sort_atoms(slab, axes=("z", "y", "x"))
    basic_tensor = torch.tensor(
        sorted_slab.get_atomic_numbers(), 
        dtype=torch.int64
    ).reshape(z_size, y_size, x_size)
    
    # 新しいテンソルのサイズ（x,y方向：2倍）
    new_x_size = 2 * x_size
    new_y_size = 2 * y_size
    
    # z はそのまま
    #interleaved = torch.zeros((z_size, new_y_size, new_x_size), dtype=torch.int64)
    # z はそのまま, zeros の代わりに full を使用して任意の初期値を設定
    interleaved = torch.full((z_size, new_y_size, new_x_size), fill_value=0.0, dtype=torch.float64)

    # 各 z 層に対して、パターンを設定
    for z in range(z_size):
        if z % 2 == 0:
            # 偶数 z 層（人間の1層目，3層目…）：
            # 基本テンソルの行 i は、interleaved の行 2*i に配置し、
            # 列も偶数インデックス (0,2,4,...)
            interleaved[z, 0::2, 0::2] = basic_tensor[z, :, :]
        else:
            # 奇数 z 層（人間の2層目，4層目…）：
            # 基本テンソルの行 i は、interleaved の行 2*i+1 に配置し、
            # 列も奇数インデックス (1,3,5,...)
            interleaved[z, 1::2, 1::2] = basic_tensor[z, :, :]
    
    return interleaved


def tensor_to_structure(tensor, template_slab):
    """
    interleaved 状態のテンソルから結晶構造（ASE Atoms オブジェクト）を復元します。
    slab_to_tensor の場合、各 z 層について、
      ・z 層が偶数なら interleaved[z, 0::2, 0::2] の要素が元の原子番号
      ・z 層が奇数なら interleaved[z, 1::2, 1::2] の要素が元の原子番号
    となっているので、それぞれ抽出して元の順序（(z, y, x)）に reshape します。
    
    Parameters:
      tensor (torch.Tensor): 3 次元テンソル、shape は (z, new_y, new_x) with new_y = 2*y, new_x = 2*x
      template_slab (ase.Atoms): 復元先のテンプレート（元の結晶構造、ソート済みのもの）
      
    Returns:
      new_slab (ase.Atoms): tensor の情報を反映して復元された結晶構造
    """
    import torch
    import numpy as np
    
    z_size, new_y_size, new_x_size = tensor.shape
    
    # 元の y, x サイズを復元（新サイズは2倍なので）
    y_size = new_y_size // 2
    x_size = new_x_size // 2
    total_atoms = z_size * y_size * x_size
    
    if total_atoms != len(template_slab):
        raise ValueError("テンソルから復元する原子数と template_slab の原子数が一致しません")
    
    # 用いるリストを各 z 層毎に作成
    reconstructed = []
    for z in range(z_size):
        if z % 2 == 0:
            # 偶数 z 層：抽出は interleaved[z, 0::2, 0::2]
            layer = tensor[z, 0::2, 0::2]
        else:
            # 奇数 z 層：抽出は interleaved[z, 1::2, 1::2]
            layer = tensor[z, 1::2, 1::2]
        # layer は shape (y_size, x_size)
        reconstructed.append(layer.flatten())
    
    # 連結して (z*y_size*x_size,) の 1D 配列にする
    new_atomic_nums = torch.cat(reconstructed).numpy()
    
    new_slab = template_slab.copy()
    new_slab.set_atomic_numbers(new_atomic_nums)
    
    return new_slab
