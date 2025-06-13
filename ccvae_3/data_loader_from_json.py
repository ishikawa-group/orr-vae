import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ase.db import connect
from typing import List, Union, Dict, Any

from tool import sort_atoms, slab_to_tensor


class OrrCatalystDataset(Dataset):
    """触媒構造と過電圧データセット（二値分類ラベル対応）"""
    def __init__(self, structures_db_paths, overpotentials_json_paths, grid_size=[4, 4, 4], 
                 use_binary_labels=True, normalize_target=False):
        self.grid_size = grid_size
        self.use_binary_labels = use_binary_labels
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
        self.raw_overpotentials = []  # 元の過電圧値を保存
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
                    self.raw_overpotentials[idx] = eta
                    self.source_info[uid] = entry  # 更新
                else:
                    # 新規追加
                    self.valid_indices.append(uid)
                    self.raw_overpotentials.append(eta)
                    self.source_info[uid] = entry
        
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
            print(f"  標準偏差: {self.overpotential_std:.3f} V")
        
        # ラベルの作成
        if self.use_binary_labels:
            # 二値分類ラベル: 中央値以上なら0（高過電圧=低性能）、未満なら1（低過電圧=高性能）
            self.targets = []
            high_performance_count = 0
            for eta in self.raw_overpotentials:
                if eta < self.overpotential_median:
                    self.targets.append(1)  # 高性能（低過電圧）
                    high_performance_count += 1
                else:
                    self.targets.append(0)  # 低性能（高過電圧）
            
            print(f"二値分類ラベル統計:")
            print(f"  高性能触媒（ラベル1）: {high_performance_count}個")
            print(f"  低性能触媒（ラベル0）: {len(self.targets) - high_performance_count}個")
            print(f"  閾値（中央値）: {self.overpotential_median:.3f} V")
            
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
        slab = self.structures[uid]
        target = self.targets[idx]
        
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
        
        # ラベルの型を適切に設定
        if self.use_binary_labels:
            target_tensor = torch.tensor(target, dtype=torch.long)  # 分類用
        else:
            target_tensor = torch.tensor(target, dtype=torch.float32)  # 回帰用
            
        return result, target_tensor
    
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
                           use_binary_labels=True, normalize_target=False):
    """複数のJSONファイルからデータセットを作成"""
    dataset = CatalystOrrDataset(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size,
        use_binary_labels=use_binary_labels,
        normalize_target=normalize_target
    )
    return dataset

def make_data_loaders_from_json(structures_db_paths, overpotentials_json_paths, 
                               train_ratio=0.9, batch_size=4, 
                               num_workers=0, seed=42,
                               grid_size=[4, 4, 4],
                               use_binary_labels=True, normalize_target=False):
    """複数のJSONファイルからデータローダーを作成"""
    dataset = create_dataset_from_json(
        structures_db_paths=structures_db_paths,
        overpotentials_json_paths=overpotentials_json_paths,
        grid_size=grid_size,
        use_binary_labels=use_binary_labels,
        normalize_target=normalize_target
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