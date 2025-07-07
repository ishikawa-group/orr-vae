import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ase.db import connect
from typing import List, Union, Dict, Any


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


def structure_to_tensor(structure, grid_size):
    """
    結晶構造を、グリッドサイズ [x, y, z] に基づいて 3 次元テンソルに変換します。
    
    ※まず、ソート（axes=("z", "y", "x")）により (z, y, x) の形状に reshape し、
      その後、x・y 方向に交互に 0 を挿入します。
    ※さらに、各 z 層で、偶数層（z index even）の場合は基本パターン（even行・even列）、
      奇数層（z index odd）の場合はシフトしたパターン（odd行・odd列）に配置します。
    
    Parameters:
      structure (ase.Atoms): 変換対象の結晶構造（原子数は x*y*z と一致）
      grid_size (list or tuple): [x, y, z]（例: [8, 8, 3]）
      
    Returns:
      tensor (torch.Tensor): interleaved な 3 次元テンソル
        最終の shape は (z, new_y, new_x) で new_y = 2*y, new_x = 2*x
    """
    import torch
    import numpy as np
    
    x_size, y_size, z_size = grid_size  # grid_size = [x, y, z]
    total_cells = x_size * y_size * z_size
    
    if len(structure) != total_cells:
        raise ValueError(f"結晶内の原子数 {len(structure)} がグリッドセル数 {total_cells} と一致しません")
    
    # ソートして (z, y, x) の形状に reshape
    sorted_structure = sort_atoms(structure, axes=("z", "y", "x"))
    basic_tensor = torch.tensor(
        sorted_structure.get_atomic_numbers(), 
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


def tensor_to_structure(tensor, template_structure):
    """
    interleaved 状態のテンソルから結晶構造（ASE Atoms オブジェクト）を復元します。
    structure_to_tensor の場合、各 z 層について、
      ・z 層が偶数なら interleaved[z, 0::2, 0::2] の要素が元の原子番号
      ・z 層が奇数なら interleaved[z, 1::2, 1::2] の要素が元の原子番号
    となっているので、それぞれ抽出して元の順序（(z, y, x)）に reshape します。
    
    Parameters:
      tensor (torch.Tensor): 3 次元テンソル、shape は (z, new_y, new_x) with new_y = 2*y, new_x = 2*x
      template_structure (ase.Atoms): 復元先のテンプレート（元の結晶構造、ソート済みのもの）
      
    Returns:
      new_structure (ase.Atoms): tensor の情報を反映して復元された結晶構造
    """
    import torch
    import numpy as np
    
    z_size, new_y_size, new_x_size = tensor.shape
    
    # 元の y, x サイズを復元（新サイズは2倍なので）
    y_size = new_y_size // 2
    x_size = new_x_size // 2
    total_atoms = z_size * y_size * x_size
    
    if total_atoms != len(template_structure):
        raise ValueError("テンソルから復元する原子数と template_structure の原子数が一致しません")
    
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
    
    new_structure = template_structure.copy()
    new_structure.set_atomic_numbers(new_atomic_nums)
    
    return new_structure


class CatalystOrrDataset(Dataset):
    """触媒構造と過電圧データセット（二値分類ラベル対応）"""

    def __init__(self, structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4],
                 use_binary_labels=True, normalize_target=False, top_n_high_performance=64,
                 top_n_low_pt_fraction=64):
        self.grid_size = grid_size
        self.use_binary_labels = use_binary_labels
        self.normalize_target = normalize_target
        self.top_n_high_performance = top_n_high_performance
        self.top_n_low_pt_fraction = top_n_low_pt_fraction
        
        # 文字列が渡された場合はリストに変換（構造DB用）
        if isinstance(structures_db_paths, str):
            structures_db_paths = [structures_db_paths]
            
        # 文字列が渡された場合はリストに変換（過電圧JSON用）
        if isinstance(overpotentials_json_paths, str):
            overpotentials_json_paths = [overpotentials_json_paths]
            
        # 全ての構造データベースからデータを読み込む（修正版：unique_idを使用）
        self.structures = {}
        for db_path in structures_db_paths:
            print(f"構造データベースを読み込み中: {db_path}")
            if not os.path.exists(db_path):
                print(f"警告: ファイルが見つかりません: {db_path}")
                continue
                
            try:
                db = connect(db_path)
                for row in db.select():
                    # 修正: unique_id属性を使用（データ生成スクリプトと同じ方法）
                    uid = row.unique_id  # 文字列IDを使用
                    self.structures[uid] = row.toatoms()
            except Exception as e:
                print(f"エラー: {db_path} の読み込みに失敗しました: {e}")
                continue
        
        print(f"合計 {len(self.structures)} 個の構造を読み込みました")
        print(f"構造IDサンプル（最初の3件）: {list(self.structures.keys())[:3]}")
        
        # 全ての過電圧JSONから結果を統合
        self.overpotentials = []
        for json_path in overpotentials_json_paths:
            print(f"過電圧データを読み込み中: {json_path}")
            if not os.path.exists(json_path):
                print(f"警告: ファイルが見つかりません: {json_path}")
                continue
                
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # リストでない場合はリストに変換
                    if not isinstance(data, list):
                        data = [data]
                    self.overpotentials.extend(data)
            except Exception as e:
                print(f"エラー: {json_path} の読み込みに失敗しました: {e}")
                continue
        
        print(f"合計 {len(self.overpotentials)} 個の過電圧データを読み込みました")
        print(f"過電圧IDサンプル（最初の3件）: {[entry.get('unique_id') for entry in self.overpotentials[:3]]}")
        
        # 有効なデータのインデックスを作成（修正版：文字列IDをそのまま使用）
        self.valid_indices = []
        self.raw_overpotentials = []  # 生の過電圧値
        self.raw_pt_fractions = []   # 生のPt割合値
        self.targets = []  # 処理後のターゲット値（2つの条件ラベル）
        self.source_info = {}  # データソース情報を保存
        
        for entry in self.overpotentials:
            uid = entry.get('unique_id')
            # 修正: 文字列IDをそのまま使用（数値変換しない）
            eta = entry.get('overpotential')
            pt_frac = entry.get('pt_fraction')
            
            # 構造、η、pt_fractionの全てが存在する場合のみ有効
            if uid in self.structures and eta is not None and pt_frac is not None:
                # 既に同じIDがある場合は後のデータで上書き
                if uid in self.source_info:
                    idx = self.valid_indices.index(uid)
                    self.raw_overpotentials[idx] = eta
                    self.raw_pt_fractions[idx] = pt_frac
                    self.source_info[uid] = entry  # 更新
                else:
                    # 新規追加
                    self.valid_indices.append(uid)
                    self.raw_overpotentials.append(eta)
                    self.raw_pt_fractions.append(pt_frac)
                    self.source_info[uid] = entry
        
        print(f"マッチしたデータ数: {len(self.valid_indices)}")
        
        # 過電圧の統計情報を計算
        if self.raw_overpotentials:
            self.overpotential_median = np.median(self.raw_overpotentials)
            self.overpotential_mean = np.mean(self.raw_overpotentials)
            self.overpotential_std = np.std(self.raw_overpotentials)
            self.overpotential_min = min(self.raw_overpotentials)
            self.overpotential_max = max(self.raw_overpotentials)
            
            print(f"過電圧統計:")
            print(f"  範囲: {self.overpotential_min:.3f} ~ {self.overpotential_max:.3f} V")
            print(f"  平均: {self.overpotential_mean:.3f} V")
            print(f"  中央値: {self.overpotential_median:.3f} V")
        else:
            print("警告: 有効なデータが見つかりませんでした")
            raise ValueError("データセットが空です。入力ファイルを確認してください。")
        
        # Pt割合の統計情報を計算
        if self.raw_pt_fractions:
            self.pt_fraction_median = np.median(self.raw_pt_fractions)
            self.pt_fraction_mean = np.mean(self.raw_pt_fractions)
            self.pt_fraction_std = np.std(self.raw_pt_fractions)
            self.pt_fraction_min = min(self.raw_pt_fractions)
            self.pt_fraction_max = max(self.raw_pt_fractions)
            
            print(f"Pt割合統計:")
            print(f"  範囲: {self.pt_fraction_min:.3f} ~ {self.pt_fraction_max:.3f}")
            print(f"  平均: {self.pt_fraction_mean:.3f}")
            print(f"  中央値: {self.pt_fraction_median:.3f}")
        
        # 2つの条件ラベルの作成
        if self.use_binary_labels:
            # 中央値を使用してラベルを決定
            # 過電圧ラベル: 中央値未満なら1（高性能）、以上なら0
            overpotential_labels = [1 if eta < self.overpotential_median else 0 for eta in self.raw_overpotentials]
            
            # Pt割合ラベル: 中央値未満なら1（低Pt）、以上なら0
            pt_fraction_labels = [1 if frac < self.pt_fraction_median else 0 for frac in self.raw_pt_fractions]

            # 2つのラベルを結合
            self.targets = [[overpotential_labels[i], pt_fraction_labels[i]] for i in range(len(self.raw_overpotentials))]

            # ラベル統計情報の表示
            high_performance_count = sum(overpotential_labels)
            low_pt_count = sum(pt_fraction_labels)

            print(f"二値分類ラベル統計 (中央値基準):")
            print(f"  高性能触媒（過電圧 < {self.overpotential_median:.3f} V）: {high_performance_count}個")
            print(f"  低性能触媒（過電圧 >= {self.overpotential_median:.3f} V）: {len(overpotential_labels) - high_performance_count}個")
            print(f"  低Pt割合触媒（Pt割合 < {self.pt_fraction_median:.3f}）: {low_pt_count}個")
            print(f"  高Pt割合触媒（Pt割合 >= {self.pt_fraction_median:.3f}）: {len(pt_fraction_labels) - low_pt_count}個")
            
        else:
            # 従来の連続値（正規化可能）
            self.targets = self.raw_overpotentials.copy()
            if normalize_target:
                if abs(self.overpotential_max - self.overpotential_min) < 1e-6:
                    self.overpotential_min = self.overpotential_min - 0.5
                    self.overpotential_max = self.overpotential_max + 0.5
                # 正規化
                self.targets = [
                    (eta - self.overpotential_min) / (self.overpotential_max - self.overpotential_min)
                    for eta in self.targets
                ]
                print(f"連続値ラベル（正規化済み）を使用")
            else:
                print(f"連続値ラベル（元の過電圧値）を使用")
        
        print(f"有効なデータ数: {len(self.valid_indices)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        uid = self.valid_indices[idx]
        structure = self.structures[uid]
        target = self.targets[idx]
        
        # 構造をテンソルに変換
        atoms_sorted = sort_atoms(structure, axes=("z", "y", "x"))
        structure_tensor = structure_to_tensor(atoms_sorted, self.grid_size)
        
        # 0（未使用位置）はそのままに、原子番号をID値にマッピング
        result = torch.zeros_like(structure_tensor, dtype=torch.long)
        
        # Ni(28)の位置を1にマッピング（Ni-Pt合金対応）
        ni_mask = (structure_tensor == 28)
        result[ni_mask] = 1
        
        # Pt(78)の位置を2にマッピング
        pt_mask = (structure_tensor == 78)
        result[pt_mask] = 2
        
        # 2つの条件ラベルを返す（[overpotential_label, pt_fraction_label]）
        if self.use_binary_labels:
            target_tensor = torch.tensor(target, dtype=torch.float32)  # [2]のテンソル
        else:
            # 連続値の場合は従来通り
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
        return result, target_tensor
    
    def get_target_range(self):
        """正規化に使用した範囲を取得"""
        return self.overpotential_min, self.overpotential_max
    
    def get_overpotential_stats(self):
        """過電圧の統計情報を取得"""
        return {
            'median': self.overpotential_median,
            'mean': self.overpotential_mean,
            'std': self.overpotential_std,
            'min': self.overpotential_min,
            'max': self.overpotential_max
        }
    
    def get_raw_overpotential(self, idx):
        """指定したインデックスの元の過電圧値を取得"""
        return self.raw_overpotentials[idx]
    
    def denormalize_target(self, normalized_value):
        """正規化された値を元のスケールに戻す（連続値の場合のみ）"""
        if not self.use_binary_labels and self.normalize_target:
            return normalized_value * (self.overpotential_max - self.overpotential_min) + self.overpotential_min
        else:
            return normalized_value

def create_dataset_from_json(structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4],
                           use_binary_labels=True, normalize_target=False, top_n_high_performance=64):
    """複数のJSONファイルからデータセットを作成"""
    dataset = CatalystOrrDataset(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size,
        use_binary_labels=use_binary_labels,
        normalize_target=normalize_target,
        top_n_high_performance=top_n_high_performance
    )
    return dataset

def make_data_loaders_from_json(structures_db_paths, overpotentials_json_paths,
                               train_ratio=0.9, batch_size=4,
                               num_workers=0, seed=42,
                               grid_size=[4, 4, 4],
                               use_binary_labels=True, normalize_target=False,
                               top_n_high_performance=64, top_n_low_pt_fraction=64):
    """複数のJSONファイルからデータローダーを作成（2つの条件ラベル対応）"""

    print(f"データローダーを作成中...")
    print(f"構造DB: {structures_db_paths}")
    print(f"過電圧JSON: {overpotentials_json_paths}")
    print(f"グリッドサイズ: {grid_size}")
    print(f"二値分類ラベル: {use_binary_labels}")
    print(f"高性能触媒（過電圧）: {top_n_high_performance}個")
    print(f"低Pt割合触媒: {top_n_low_pt_fraction}個")
    
    dataset = CatalystOrrDataset(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size,
        use_binary_labels=use_binary_labels,
        normalize_target=normalize_target,
        top_n_high_performance=top_n_high_performance,
        top_n_low_pt_fraction=top_n_low_pt_fraction  # 新しいパラメータを渡す
    )
    
    if len(dataset) == 0:
        raise ValueError("データセットが空です。入力ファイルを確認してください。")
    
    # データセットの分割
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    
    print(f"データ分割: 訓練={n_train}, テスト={n_test}")
    
    if n_train == 0 or n_test == 0:
        print("警告: 訓練またはテストデータが0件です。train_ratioを調整してください。")
    
    g = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=g)
    
    # データローダーの作成
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
    
    print(f"データローダー作成完了: batch_size={batch_size}")
    
    return train_loader, test_loader, dataset
