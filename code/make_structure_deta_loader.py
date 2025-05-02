from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ase.db import connect

# tool.py のユーティリティをインポート
from ORR_catalyst_generator.code.tool import sort_atoms, slab_to_tensor

# デフォルト設定
GRID_SIZE = (4, 4, 4)
BATCH_SIZE = 16
NUM_WORKERS = 4

class CatalystDataset(Dataset):
    """
    (slab_tensor, overpotential) ペアを返すデータセット
    """
    def __init__(self, struct_json, energy_json, grid_size=GRID_SIZE):
        self.grid_size = grid_size

        # 1. ラベル（過電圧）を読む
        with open(energy_json, "r", encoding="utf-8") as f:
            energy_records = json.load(f)

        self.id_list = [rec["unique_id"] for rec in energy_records]
        self.labels = {
            rec["unique_id"]:
                float(Decimal(rec["overpotential"])
                      .quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))  # 小数点第3位まで丸める
            for rec in energy_records
        }

        # 2. 構造 DB を開く
        self.db = connect(str(struct_json))

    def __len__(self):
        return len(self.id_list)

    # 内部ヘルパー：unique_id → ASE Atoms 取得
    def _fetch_atoms(self, uid):
        row = None
        for query in (
            lambda: list(self.db.select(f'unique_id="{uid}"')),
            lambda: list(self.db.select(unique_id=uid)),
            lambda: [r for r in self.db.select()
                     if r.key_value_pairs.get("unique_id") == uid]
        ):
            try:
                rows = query()
                if rows:
                    row = rows[0]
                    break
            except (ValueError, KeyError):
                continue
                
        if row is None:
            raise KeyError(f"[CatalystDataset] unique_id {uid} not found")

        atoms = row.toatoms(add_additional_information=True)
        if hasattr(atoms, 'info') and 'data' in atoms.info:
            info = atoms.info.pop("data", {})
            atoms.info["adsorbate_info"] = info.get("adsorbate_info", {})
        return atoms

    def __getitem__(self, idx):
        uid = self.id_list[idx]
        overpot = self.labels[uid]

        atoms_sorted = sort_atoms(self._fetch_atoms(uid), axes=("z", "y", "x"))
        slab_tensor = slab_to_tensor(atoms_sorted, self.grid_size).to(dtype=torch.float32)

        return slab_tensor, torch.tensor(overpot, dtype=torch.float32)

def make_data_loaders(struct_json, energy_json, 
                     grid_size=GRID_SIZE,
                     train_ratio=0.9, 
                     batch_size=BATCH_SIZE, 
                     num_workers=NUM_WORKERS,
                     seed=42):
    """
    触媒データセットのローダーを作成
    
    Parameters
    ----------
    struct_json : Path
        構造JSONファイルへのパス
    energy_json : Path
        エネルギーJSONファイルへのパス
    grid_size : tuple, default (4, 4, 4)
        スラブ構造のグリッドサイズ
    train_ratio : float, default 0.9
        トレーニングデータの割合
    batch_size : int, default 16
        バッチサイズ
    num_workers : int, default 4
        データローディングに使用するワーカー数
    seed : int, default 42
        データ分割のための乱数シード
    
    Returns
    -------
    tuple
        (train_loader, test_loader)
    """
    dataset = CatalystDataset(struct_json=struct_json,
                             energy_json=energy_json,
                             grid_size=grid_size)
    
    # train / test にランダム分割
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    g = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=g)
    
    # DataLoaderの作成
    train_loader = DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, test_loader