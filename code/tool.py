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


def parallel_displacement(atoms, vacuum=15.0):
    """
    スラブを z 軸方向に平行移動させ、最低点が z=0 になるようにし、
    指定された真空層 (vacuum[Å]) を上側（z正方向）に追加する関数です。
    
    注意:
        - この関数は、入力のスラブが表面法線方向に z 軸が一致していることを前提とします。
        - すでに斜交セル等の場合は、予め回転などの前処理を行ってください。
    
    Args:
        atoms: ASE Atoms オブジェクト（スラブ。vacuumオプションなしで生成したものが望ましい）
        vacuum: 追加する真空層の厚さ (Å)。デフォルトは 15.0 Å。
    
    Returns:
        原子位置を z=0 に下詰めし、セルの z 軸長を (スラブの高さ + vacuum) に設定した
        新しい ASE Atoms オブジェクト。
    """
    # 元のオブジェクトを変更しないようにコピーを作成
    slab = atoms.copy()

    # 現在の原子位置を取得し、z方向の最小値を計算
    positions = slab.get_positions()
    zmin = positions[:, 2].min()

    # スラブ全体を z 軸方向に平行移動し、最低点が z=0 になるようにする
    slab.translate([0, 0, -zmin])

    # 平行移動後の最高 z 座標を取得
    zmax = slab.get_positions()[:, 2].max()
    # 新しいセルの z 軸長（スラブ高さ + vacuum）を計算
    new_z_length = zmax + vacuum

    # セル行列を取得して z 軸方向のサイズを新しい長さにセットする
    # ※ここでは、セルの第3ベクトルが z 軸方向に並んでいる前提
    cell = slab.get_cell().copy()
    # 安全のため、z 軸の成分を [0, 0, new_z_length] に再設定する方法もあります
    cell[2] = [0.0, 0.0, new_z_length]
    slab.set_cell(cell, scale_atoms=False)  # scale_atoms=False で原子座標は変更せずセルだけ更新

    return slab

def auto_lmaxmix(atoms):
    """d/f 元素を含む場合 lmaxmix を自動設定"""

    d_elems = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
               "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
               "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
    f_elems = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
               "Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np",
               "Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"}
    symbs   = set(atoms.get_chemical_symbols())
    atoms.calc.set(lmaxmix = 6 if symbs & f_elems else 4 if symbs & d_elems else 2)

    return atoms

def my_calculator(
        atoms, kind:str, 
        calc_type:str="mattersim", 
        yaml_path:str="/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/code/vasp.yaml",
        calc_directory:str="calc"): 
    """
    Create calculator instance based on parameters from YAML file and attach to atoms.

    Args:
        atoms: ASE atoms ocject
        kind: "gas" / "slab" / "bulk"
        calc_type: "vasp" / "mattersim" - calculator type
        calc_directory: Calculation directory for vasp

    Returns:
        atoms: 計算機が設定されたAtomsオブジェクト（bulkの場合はFrechetCellFilter）
    """
    # すべてのインポートを関数内に配置
    import yaml
    import sys
    from typing import Dict, Any

    if calc_type.lower() == "vasp":
        from ase.calculators.vasp import Vasp
 
        # YAMLファイルを直接読み込む
        yaml_path = yaml_path
        try:
            with open(yaml_path, 'r') as f:
                vasp_params = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: VASP parameter file not found at {yaml_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            sys.exit(1)
        
        if kind not in vasp_params['kinds']:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {list(vasp_params['kinds'].keys())}")

        # 共通パラメータをコピー
        params = vasp_params['common'].copy()

        # kind固有のパラメータで更新
        params.update(vasp_params['kinds'][kind])

        # 関数引数で指定されたパラメータを設定
        params['directory'] = calc_directory

        # kptsをタプルに変換 (ASEはタプルを期待するため)
        if 'kpts' in params and isinstance(params['kpts'], list):
            params['kpts'] = tuple(params['kpts'])

        # 原子オブジェクトに計算機を設定して返す
        atoms.calc = Vasp(**params)
        # 自動的にlmaxmixを設定
        atoms = auto_lmaxmix(atoms)

    elif calc_type.lower() == "mattersim":
        # MatterSimを使用する場合
        import torch
        from mattersim.forcefield.potential import MatterSimCalculator
        from ase.filters import FrechetCellFilter, ExpCellFilter
        from ase.constraints import FixSymmetry
        from ase.optimize import FIRE, LBFGS
        
        if torch.cuda.is_available():
            device = "cuda"
        #elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        #    device = "mps"
        else:
            device = "cpu"
        atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
        
        # bulk計算の場合はCellFilterを適用
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)
        
        #構造最適化の実行
        opt = FIRE(atoms)
        opt.run(fmax=0.05, steps=300)

    elif calc_type.lower() == "sevennet":
        # MatterSimを使用する場合
        import torch
        from sevenn.calculator import SevenNetCalculator
        from ase.filters import FrechetCellFilter, ExpCellFilter
        from ase.constraints import FixSymmetry
        from ase.optimize import FIRE, LBFGS
        
        if torch.cuda.is_available():
            device = "cuda"
        #elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        #    device = "mps"
        else:
            device = "cpu"
        atoms.calc = SevenNetCalculator('7net-mf-ompa', modal='mpa', device=device)
        
        # bulk計算の場合はExpCellFilterを適用
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)
        
        #構造最適化の実行
        opt = FIRE(atoms)
        opt.run(fmax=0.05, steps=200)
        
    else:
        raise ValueError("calc_type must be 'vasp' or 'mattersim'")
    
    return atoms


def set_initial_magmoms(atoms, kind:str="bulk", formula:str=None):
    """
    原子に初期磁気モーメントを設定する関数
    
    Args:
        atoms: ASE atoms オブジェクト
        kind: "gas" / "slab" / "bulk" - 系の種類
        formula: 分子式 (kindが"gas"の場合に使用)
        
    Returns:
        atoms: 磁気モーメントが設定されたAtomsオブジェクト
    """
    # 定数を関数内で定義
    MAG_ELEMENTS = ["Mn", "Fe", "Cr"]  # 初期磁気モーメント 1.0 μB
    CLOSED_SHELL = ["H2", "H2O"]       # スピン非分極で計算する分子
    
    symbols = atoms.get_chemical_symbols()
    
    # gas相で閉殻分子の場合は全て0に
    if kind == "gas" and formula in CLOSED_SHELL:
        init_magmom = [0.0] * len(symbols)
    else:
        # 磁性元素には1.0 μB, それ以外は0.0を設定
        init_magmom = [1.0 if x in MAG_ELEMENTS else 0.0 for x in symbols]
    
    atoms.set_initial_magnetic_moments(init_magmom)
    return atoms  # 変更後のatomsを返す

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
