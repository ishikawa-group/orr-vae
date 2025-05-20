import os
import sys
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------------
# 触媒データローダーの実装
# ------------------------------
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP

from torch.utils.data import Dataset, DataLoader, random_split
from ase.db import connect

# tool.py のユーティリティをインポート
from tool import sort_atoms, slab_to_tensor

# データローダーのデフォルト設定
GRID_SIZE = (4, 4, 4)  # original slab shape (x, y, z)
BATCH_SIZE = 16
NUM_WORKERS = 4

class CatalystDataset(Dataset):
    """(slab_tensor, overpotential) ペアを返すデータセット"""
    def __init__(self, struct_json, energy_json, grid_size=GRID_SIZE):
        self.grid_size = grid_size

        # 1. ラベル（過電圧）を読み込む
        with open(energy_json, "r", encoding="utf-8") as f:
            energy_records = json.load(f)

        self.id_list = [rec["unique_id"] for rec in energy_records]
        self.labels = {
            rec["unique_id"]:
                float(Decimal(rec["overpotential"])
                      .quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))  # 小数点第3位まで
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
        slab_tensor = slab_to_tensor(atoms_sorted, self.grid_size)
        
        # 0（未使用位置）はそのままに、原子番号をID値にマッピング
        result = torch.zeros_like(slab_tensor, dtype=torch.long)  # すべて0.0で初期化
        
        # Pd(46)の位置を0にマッピング
        pd_mask = (slab_tensor == 46)
        result[pd_mask] = 1.0
        
        # Pt(78)の位置を1にマッピング
        pt_mask = (slab_tensor == 78)
        result[pt_mask] = 2.0
        
        return result, torch.tensor(overpot, dtype=torch.float32)

def make_data_loaders(struct_json, energy_json, 
                     grid_size=GRID_SIZE,
                     train_ratio=0.9, 
                     batch_size=BATCH_SIZE, 
                     num_workers=NUM_WORKERS,
                     seed=42):
    """触媒データセットのローダーを作成"""
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

# ------------------------------
# ハイパーパラメータの設定
# ------------------------------
batch_size    = 16
learning_rate = 1e-3
max_epoch     = 30
num_workers   = 4
load_epoch    = -1     # ロードするエポック（-1なら新規学習）

# 潜在変数の次元
latent_size   = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ------------------------------
# モデル定義（条件付きVAE - 連続値対応版）
# ------------------------------
class CVAE(nn.Module):
    def __init__(self, latent_size=64, condition_dim=1):
        """
        latent_size : 潜在変数の次元数
        condition_dim : 条件の次元数（overpotentialの連続値は1次元）
        """
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.condition_dim = condition_dim
        self.activation = nn.SiLU()

        # Encoder
        # 入力はスラブ構造 (Z=4層) と条件値（チャネルとして結合）
        self.conv1   = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.linear1 = nn.Linear(128 * 2 * 2, 256)
        self.mu      = nn.Linear(256, self.latent_size)
        self.logvar  = nn.Linear(256, self.latent_size)

        # 条件値の非線形変換
        self.label_fc1 = nn.Linear(condition_dim, 64)
        self.label_fc2 = nn.Linear(64, 16)

        # Decoder
        self.linear2 = nn.Linear(self.latent_size + 16, 256)
        self.linear3 = nn.Linear(256, 64 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=4*3, kernel_size=3, stride=1, padding=1)

    def encoder(self, x, y):
        """
        x : 入力スラブテンソル [B, 4, 8, 8]
        y : 連続値overpotential [B, 1]
        """
        # 連続値を特徴マップとして拡張
        y_expanded = y.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        # チャネル方向に結合（Z=4のスラブ + 条件値1チャネル）
        t = torch.cat((x, y_expanded), dim=1)
        
        t = self.activation(self.bn1(self.conv1(t)))
        t = self.activation(self.bn2(self.conv2(t)))
        t = self.activation(self.bn3(self.conv3(t)))
        t = t.view(t.size(0), -1)
        t = self.activation(self.linear1(t))
        
        mu     = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return eps * std + mu

    def unflatten(self, x):
        return x.view(x.size(0), 64, 2, 2)

    def decoder(self, z):
        """
        z : 潜在変数と条件埋め込みの結合 [B, latent_size+16]
        """
        t = self.activation(self.linear2(z))
        t = self.activation(self.linear3(t))
        t = self.unflatten(t)
        t = self.activation(self.bn4(self.deconv1(t)))
        t = self.activation(self.bn5(self.deconv2(t)))
        t = self.deconv3(t)  

        # シグモイド関数の代わりに温度付きシグモイド関数を使用
        #temperature = 0.1  # 小さい温度でより鋭い分布に
        #t = torch.sigmoid(t / temperature)

        return t

    def label_encoder(self, y):
        """
        連続値を非線形変換
        y : [B, 1] (overpotentialの値)
        """
        h1 = self.activation(self.label_fc1(y))
        h2 = self.activation(self.label_fc2(h1))
        return h2

    def forward(self, x, y):
        """
        x : 入力スラブ [B, 4, 8, 8]
        y : overpotential値 [B, 1]
        """
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        
        # 連続値をエンコード
        label_embedding = self.label_encoder(y)
        
        # 潜在変数と条件埋め込みを結合
        z_cat = torch.cat((z, label_embedding), dim=1)
        pred = self.decoder(z_cat)
        return pred, mu, logvar

# ------------------------------
# 損失関数
# ------------------------------
def loss_function(x, pred, mu, logvar):
    # pred: [B, 12, H, W] - logits
    # x: [B, 4, H, W] - クラスインデックス（各層ごとに0,1,2のいずれか）
    
    x = x.to(dtype=torch.long)
    
    # 各層ごとに処理する場合
    recon_loss = 0
    for z in range(4):  # 4層
        # 各層で3クラス分類
        layer_pred = pred[:, z*3:(z+1)*3]  # 各層に3チャネル
        recon_loss += F.cross_entropy(
            layer_pred, 
            x[:, z], 
            reduction='sum'
        )
    
    # KL損失
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld

# ------------------------------
# 学習ループ
# ------------------------------
def train(epoch, model, train_loader, optimizer):
    model.train()
    total_recon, total_kld, total_loss = 0, 0, 0
    start_time = time.time()
    
    for i, (x, y) in enumerate(train_loader):
        try:
            x = x.to(device)
            # 連続値は1次元ベクトルに変換
            y = y.to(device).view(-1, 1).float()

            optimizer.zero_grad()
            pred, mu, logvar = model(x, y)
            recon_loss, kld = loss_function(x, pred, mu, logvar)
            loss = recon_loss + (epoch/max_epoch)*kld*0.1
            loss.backward()
            optimizer.step()

            total_loss  += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kld   += kld.item() * x.size(0)
            
            if i % 1 == 0:
                print(f"Batch {i}/{len(train_loader)}: Loss={loss.item():.4f}")
                
        except Exception as e:
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    dataset_size = len(train_loader.dataset)
    elapsed = time.time() - start_time
    return total_loss / dataset_size, total_kld / dataset_size, total_recon / dataset_size, elapsed

def test(epoch, model, test_loader):
    model.eval()
    total_recon, total_kld, total_loss = 0, 0, 0
    start_time = time.time()

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            try:
                x = x.to(device)
                y = y.to(device).view(-1, 1).float()

                pred, mu, logvar = model(x, y)
                recon_loss, kld = loss_function(x, pred, mu, logvar)
                loss = recon_loss + kld

                total_loss  += loss.item() * x.size(0)
                total_recon += recon_loss.item() * x.size(0)
                total_kld   += kld.item() * x.size(0)
                
                # 最終エポックのみ可視化
                if epoch == max_epoch - 1 and i == 0:
                    visualize_slabs(pred.cpu(), y.cpu(), result_dir)
                    
            except Exception as e:
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

    dataset_size = len(test_loader.dataset)
    elapsed = time.time() - start_time
    return total_loss / dataset_size, total_kld / dataset_size, total_recon / dataset_size, elapsed

# ------------------------------
# 可視化と結果保存
# ------------------------------
def visualize_slabs(pred, y, result_dir):
    """スラブ構造の可視化 - 最終結果のみ保存"""
    fig = plt.figure(figsize=(16, 12))
    
    for i in range(min(6, pred.size(0))):
        # スラブの各層を表示
        for z in range(4):
            ax = fig.add_subplot(6, 4, i*4 + z + 1)
            ax.imshow(pred[i, z].numpy(), cmap='viridis')
            ax.axis('off')
            if z == 0:
                ax.set_title(f"OP: {y[i, 0]:.3f}")
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/reconstructed_slabs.png")
    plt.close()

def generate_samples(model, result_dir):
    """潜在空間からのサンプル生成 - 最終結果のみ保存"""
    model.eval()
    with torch.no_grad():
        # 異なるoverpotential値でサンプル生成
        z = torch.randn(6, latent_size).to(device)
        y_values = torch.tensor([[0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], dtype=torch.float).to(device)
        
        # 条件と潜在変数からスラブ生成
        label_embedding = model.label_encoder(y_values)
        z_cat = torch.cat((z, label_embedding), dim=1)
        pred = model.decoder(z_cat)
        
        # 各位置で最も確率の高いクラスを選択
        # 4層×3クラスの場合
        final_output = torch.zeros(6, 4, 8, 8)
        for z in range(4):
            layer_logits = pred[:, z*3:(z+1)*3]
            probs = F.softmax(layer_logits, dim=1)
            # 最も確率の高いクラスを選択
            _, predicted = torch.max(probs, dim=1)
            
            # 表示用に値をマッピング
            viz_map = torch.zeros_like(predicted, dtype=torch.float)
            viz_map[predicted == 0] = 0.0  # 空位置
            viz_map[predicted == 1] = 1.0  # Pt
            viz_map[predicted == 2] = 2.0  # Pd
            
            final_output[:, z] = viz_map
            
            plt.tight_layout()
            plt.savefig(f"{result_dir}/generated_samples.png")
            plt.close()

def plot_learning_curves(train_loss, test_loss, result_dir):
    """学習曲線のプロット - 最終結果のみ保存"""
    epochs = np.arange(1, len(train_loss) + 1)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss[:, 0], 'b-', label='Training Total Loss')
    plt.plot(epochs, test_loss[:, 0], 'r-', label='Test Total Loss')
    plt.title('Total Loss Progress', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_loss[:, 1], 'b-', label='Training KL Loss')
    plt.plot(epochs, test_loss[:, 1], 'r-', label='Test KL Loss')
    plt.title('KL Divergence Progress', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_loss[:, 2], 'b-', label='Training Reconstruction Loss')
    plt.plot(epochs, test_loss[:, 2], 'r-', label='Test Reconstruction Loss')
    plt.title('Reconstruction Loss Progress', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/learning_curves.png")
    plt.close()

# ------------------------------
# メイン処理
# ------------------------------
if __name__ == "__main__":
    # 結果保存先を設定
    result_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE"
    os.makedirs(result_dir, exist_ok=True)
    
    # データローダーを作成
    train_loader, test_loader = make_data_loaders(
        struct_json=Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/data/iter0_structure.json"),
        energy_json=Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/data/iter0_reaction_energy.json"),
        grid_size=(4, 4, 4),
        num_workers=num_workers,
        train_ratio=0.9,
        batch_size=batch_size
    )
    print("データローダーを作成しました")
    
    # モデル初期化
    model = CVAE(latent_size=latent_size, condition_dim=1).to(device)
    print("モデルを作成しました")
    
    # 既存モデルのロード（必要な場合）
    if load_epoch > 0:
        checkpoint_path = f'./checkpoints/model_{load_epoch}.pt'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"モデル {load_epoch} をロードしました")
    
    # オプティマイザ設定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
    
    # 損失履歴を保存するリスト
    train_loss_list = []
    test_loss_list = []
    
    # 学習ループ
    for epoch in range(load_epoch + 1, max_epoch):
        train_total, train_kld, train_recon, train_time = train(epoch, model, train_loader, optimizer)
        test_total, test_kld, test_recon, test_time = test(epoch, model, test_loader)
        
        print(f"Epoch: {epoch+1}/{max_epoch} | 損失: {train_total:.4f} | 学習時間: {train_time:.2f}秒 / 評価時間: {test_time:.2f}秒")
        
        # 損失履歴を追加
        train_loss_list.append([train_total, train_kld, train_recon])
        test_loss_list.append([test_total, test_kld, test_recon])
        
        # 最終エポックのみモデル保存
        if epoch == max_epoch - 1:
            torch.save(model.state_dict(), f"{result_dir}/final_model.pt")
    
    # 最終結果の保存
    train_loss_array = np.array(train_loss_list)
    test_loss_array = np.array(test_loss_list)
    
    # 学習曲線をプロット・保存
    plot_learning_curves(train_loss_array, test_loss_array, result_dir)
    
    # 生成サンプルを保存
    generate_samples(model, result_dir)
    
    print(f"学習結果は {result_dir} に保存されました")