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
    tags = pd.cut(zpos, bins=bins, labels=labels, include_lowest=True).to_list()
    newatoms.set_tags(tags)
    
    return newatoms


def fix_lower_surface(atoms):
    """
    モデルの下半分の層を固定する関数。
    まず原子に z 座標に基づいたタグを設定し、
    その後、下半分の層に属する原子を固定する。
    
    例: 3層の場合は floor(3/2)=1 となり、下層1層分が固定される。
    
    Args:
        atoms: ASE atoms オブジェクト
        
    Returns:
        下半分が固定された新しい atoms オブジェクト
    """
    import numpy as np
    from ase.constraints import FixAtoms

    atom_fix = atoms.copy()

    # タグ付け（下層からの階層番号）
    atom_fix = set_tags_by_z(atom_fix)
    # タグ情報を取得
    tags = atom_fix.get_tags()

    # 全体の層数を取得（get_number_of_layers 内の丸め精度と set_tags_by_z の丸め精度は用途に合わせて調整）
    nlayer = get_number_of_layers(atom_fix)
    # 下半分の層番号（端数は切り捨て）
    lower_layers = list(range(nlayer // 2))
    
    # 固定対象の原子インデックスを選択
    fix_indices = [atom.index for atom in atom_fix if atom.tag in lower_layers]
    
    # FixAtoms 制約を適用
    c = FixAtoms(indices=fix_indices)
    atom_fix.set_constraint(c)

    return atom_fix

def get_overpotential_orr(deltaEs, T=298.15, energy_shift=None, verbose=False):
    """
    ORRの過電圧を計算するための関数
    
    参考:https://github.com/ishikawa-group (installして使いたい)
    
    使用する反応エネルギーは以下の4段階（deltaE2～deltaE5）:
      (2) ΔE2: O2* + H+ + e- → OOH*
      (3) ΔE3: OOH* + H+ + e- → O* + H2O(g)
      (4) ΔE4: O* + H+ + e- → OH*
      (5) ΔE5: OH* + H+ + e- → * + H2O(g)
    
    引数:
      deltaEs: [deltaE2, deltaE3, deltaE4, deltaE5]（リストまたはnumpy配列）
      T: 温度（デフォルトは298.15 K）
      energy_shift: エネルギー補正値（リストまたはnumpy配列、Noneの場合は補正なし）
      verbose: 詳細なログ出力をする場合はTrue
      
    戻り値:
      ORRの過電圧（η）[eV]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import logging

    logger = logging.getLogger(__name__)
    np.set_printoptions(formatter={"float": "{:0.2f}".format})

    # deltaEsの値が正しく定義されているかチェック
    if any(e is None for e in deltaEs):
        logger.error("deltaEsの値にNoneが含まれています。")
        return None

    # 零点エネルギー (ZPE) の定義 (単位: eV)
    zpe = {
        "H2": 0.27,
        "H2O": 0.56,
        "OHads": 0.36,
        "Oads": 0.07,
        "OOHads": 0.40,
        "O2": 0.05 * 2
    }

    # エントロピー (S) の定義 (単位: eV/K)
    S = {
        "H2": 0.41 / T,
        "H2O": 0.67 / T,
        "O2": 0.32 * 2 / T
    }

    # ORRの各反応ステップにおけるエントロピー変化 (ΔS) とZPE補正 (ΔZPE) の設定（全4段階）
    deltaSs = np.zeros(4)
    deltaZPEs = np.zeros(4)

    # ステップ1: O2* → OOH*
    deltaSs[0] = - S["O2"] - S["H2"]
    deltaZPEs[0] = zpe["OOHads"] - 0.5 * zpe["H2"] - zpe["O2"]

    # ステップ2: OOH* → O* + H2O(g)
    deltaSs[1] = S["H2O"] - 0.5 * S["H2"]
    deltaZPEs[1] = zpe["Oads"] + zpe["H2O"] - 0.5 * zpe["H2"] - zpe["OOHads"]

    # ステップ3: O* → OH*
    deltaSs[2] = - 0.5 * S["H2"]
    deltaZPEs[2] = zpe["OHads"] - 0.5 * zpe["H2"] - zpe["Oads"]

    # ステップ4: OH* → * + H2O(g)
    deltaSs[3] = S["H2O"] - 0.5 * S["H2"]
    deltaZPEs[3] = zpe["H2O"] - 0.5 * zpe["H2"] - zpe["OHads"]

    # numpy配列に変換
    deltaEs = np.array(deltaEs)
    # 反応エンタルピー変化： ΔH = ΔE + ΔZPE
    deltaHs = deltaEs + deltaZPEs
    # ギブス自由エネルギー変化： ΔG = ΔH - T * ΔS
    deltaGs = deltaHs - T * deltaSs

    # エネルギーシフトが指定されている場合は適用
    if energy_shift is not None:
        deltaGs += np.array(energy_shift)

    if verbose:
        logger.info(f"deltaGs: {deltaGs}")

    # ORRの計算
    # 平衡電位 (φ) の設定（単位: eV）
    phi = 1.0288

    # 各ステップまでの累積自由エネルギー変化の計算
    deltaGs_sum = [
        0.0,
        deltaGs[0],
        deltaGs[0] + deltaGs[1],
        deltaGs[0] + deltaGs[1] + deltaGs[2],
        deltaGs[0] + deltaGs[1] + deltaGs[2] + deltaGs[3]
    ]

    # 平衡状態下での自由エネルギー変化 (各ステップでφを加味)
    deltaGs_eq = [
        deltaGs_sum[0],
        deltaGs_sum[1] + phi,
        deltaGs_sum[2] + 2 * phi,
        deltaGs_sum[3] + 3 * phi,
        deltaGs_sum[4] + 4 * phi
    ]

    # 各ステップ間の自由エネルギー差を計算
    diffG = [
        deltaGs_eq[1] - deltaGs_eq[0],
        deltaGs_eq[2] - deltaGs_eq[1],
        deltaGs_eq[3] - deltaGs_eq[2],
        deltaGs_eq[4] - deltaGs_eq[3]
    ]

    # 最大の自由エネルギー差が過電圧となる
    eta = np.max(diffG)
    eta = np.abs(eta)  # 過電圧は正の値で表す

    if verbose:
        logger.info(f"deltaGs_sum: {deltaGs_sum}")
        logger.info(f"deltaGs_eq: {deltaGs_eq}")
        logger.info(f"diffG: {diffG}")
        logger.info(f"Calculated overpotential (η): {eta} eV")

    return eta

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


def slab_to_tensor(slab, grid_size):
    """
    スラブ構造を、グリッドサイズ [x, y, z] に基づいて 3 次元テンソルに変換します。
    
    ※まず、ソート（axes=("z", "y", "x")）により (z, y, x) の形状に reshape し、
      その後、x・y 方向に交互に 0 を挿入します。
    ※さらに、各 z 層で、偶数層（z index even）の場合は基本パターン（even行・even列）、
      奇数層（z index odd）の場合はシフトしたパターン（odd行・odd列）に配置します。
    
    Parameters:
      slab (ase.Atoms): 変換対象のスラブ構造（原子数は x*y*z と一致）
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
        raise ValueError(f"スラブ内の原子数 {len(slab)} がグリッドセル数 {total_cells} と一致しません")
    
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
    interleaved = torch.zeros((z_size, new_y_size, new_x_size), dtype=torch.int64)
    
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


def tensor_to_slab(tensor, template_slab):
    """
    interleaved 状態のテンソルからスラブ構造（ASE Atoms オブジェクト）を復元します。
    slab_to_tensor の場合、各 z 層について、
      ・z 層が偶数なら interleaved[z, 0::2, 0::2] の要素が元の原子番号
      ・z 層が奇数なら interleaved[z, 1::2, 1::2] の要素が元の原子番号
    となっているので、それぞれ抽出して元の順序（(z, y, x)）に reshape します。
    
    Parameters:
      tensor (torch.Tensor): 3 次元テンソル、shape は (z, new_y, new_x) with new_y = 2*y, new_x = 2*x
      template_slab (ase.Atoms): 復元先のテンプレート（元のスラブ構造、ソート済みのもの）
      
    Returns:
      new_slab (ase.Atoms): tensor の情報を反映して復元されたスラブ構造
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
