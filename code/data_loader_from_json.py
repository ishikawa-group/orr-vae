import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ase.db import connect
from typing import List, Union, Dict, Any

from tool import sort_atoms, slab_to_tensor


# クラスを修正して複数のJSONファイルに対応
class CatalystOrrDataset(Dataset):
    """触媒構造と過電圧データセット（複数のJSONファイルに対応）"""
    def __init__(self, structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4], normalize_target=True):
        self.grid_size = grid_size
        self.normalize_target = normalize_target
        
        # 文字列が渡された場合はリストに変換（構造DB用）
        if isinstance(structures_db_paths, str):
            structures_db_paths = [structures_db_paths]
            
        # 文字列が渡された場合はリストに変換（過電圧JSON用）
        if isinstance(overpotentials_json_paths, str):
            overpotentials_json_paths = [overpotentials_json_paths]
            
        # 全ての構造データベースからデータを読み込む
        self.structures = {}
        for db_path in structures_db_paths:
            print(f"構造データベースを読み込み中: {db_path}")
            db = connect(db_path)
            for row in db.select():
                uid = row.id  # データベースIDを使用
                self.structures[uid] = row.toatoms()
        
        print(f"合計 {len(self.structures)} 個の構造を読み込みました")
        
        # 全ての過電圧JSONから結果を統合
        self.overpotentials = []
        for json_path in overpotentials_json_paths:
            print(f"過電圧データを読み込み中: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                # リストでない場合はリストに変換
                if not isinstance(data, list):
                    data = [data]
                self.overpotentials.extend(data)
        
        print(f"合計 {len(self.overpotentials)} 個の過電圧データを読み込みました")
        
        # 有効なデータのインデックスを作成
        self.valid_indices = []
        self.targets = []
        self.source_info = {}  # データソース情報を保存
        
        for entry in self.overpotentials:
            uid = entry.get('unique_id')
            # データベースのIDとして数値に変換
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                continue
                
            eta = entry.get('overpotential')
            
            # 構造とηの両方が存在する場合のみ有効
            if uid in self.structures and eta is not None:
                # 既に同じIDがある場合は後のデータで上書き
                if uid in self.source_info:
                    idx = self.valid_indices.index(uid)
                    self.targets[idx] = eta
                    self.source_info[uid] = entry  # 更新
                else:
                    # 新規追加
                    self.valid_indices.append(uid)
                    self.targets.append(eta)
                    self.source_info[uid] = entry
        
        # ターゲット値の正規化（必要に応じて）
        if normalize_target and self.targets:
            self.target_min = min(self.targets)
            self.target_max = max(self.targets)
            # 値の範囲が狭い場合の処理
            if abs(self.target_max - self.target_min) < 1e-6:
                self.target_min = self.target_min - 0.5
                self.target_max = self.target_max + 0.5
        else:
            self.target_min = 0
            self.target_max = 1
        
        print(f"有効なデータ数: {len(self.valid_indices)}")
        
    # 以下の__len__と__getitem__は変更なし
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        uid = self.valid_indices[idx]
        slab = self.structures[uid]
        eta = self.targets[idx]
        
        # 構造をテンソルに変換
        atoms_sorted = sort_atoms(slab, axes=("z", "y", "x"))
        slab_tensor = slab_to_tensor(atoms_sorted, self.grid_size)
        
        # 0（未使用位置）はそのままに、原子番号をID値にマッピング
        result = torch.zeros_like(slab_tensor, dtype=torch.long)
        
        # Pd(46)の位置を1にマッピング
        pd_mask = (slab_tensor == 46)
        result[pd_mask] = 1
        
        # Pt(78)の位置を2にマッピング
        pt_mask = (slab_tensor == 78)
        result[pt_mask] = 2
        
        # 過電圧値を正規化（必要な場合）
        if self.normalize_target:
            eta_normalized = (eta - self.target_min) / (self.target_max - self.target_min)
        else:
            eta_normalized = eta
            
        return result, torch.tensor(eta_normalized, dtype=torch.float32)

# 関数を修正して複数のJSONファイルに対応
def create_dataset_from_json(structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4]):
    """複数のJSONファイルからデータセットを作成"""
    dataset = CatalystOrrDataset(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size
    )
    return dataset

def make_data_loaders_from_json(structures_db_paths, overpotentials_json_paths, 
                               train_ratio=0.9, batch_size=4, 
                               num_workers=0, seed=42,
                               grid_size=[4, 4, 4]):
    """複数のJSONファイルからデータローダーを作成"""
    dataset = create_dataset_from_json(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size
    )
    
    # データセットの分割
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    
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
    
    return train_loader, test_loader, dataset