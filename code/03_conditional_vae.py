import os
import sys
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import freeze_support

from tool import make_data_loaders_from_json

# ------------------------------
# コマンドライン引数の解析
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Conditional VAE Training')
    parser.add_argument('--iter', type=int, default=1, 
                       help='Iteration number (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--max_epoch', type=int, default=100,
                       help='Maximum epochs (default: 100)')
    parser.add_argument('--latent_size', type=int, default=32,
                       help='Latent space dimension (default: 32)')
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Beta for KL loss (default: 2.0)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Training data ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--load_epoch', type=int, default=-1,
                       help='Load epoch (-1 for no loading, default: -1)')
    parser.add_argument('--base_data_path', type=str,
                       default=str(Path(__file__).parent / "data"),
                       help='Base data directory path')
    parser.add_argument('--result_base_path', type=str,
                       default=str(Path(__file__).parent / "result"),
                       help='Base result directory path')
    return parser.parse_args()

# ------------------------------
# ハイパーパラメータの設定
# ------------------------------
# モジュールが直接実行される場合のみargparseを実行
if __name__ == "__main__":
    args = parse_args()
else:
    # 他のスクリプトからインポートされる場合はデフォルト値を使用
    class DefaultArgs:
        iter = 1
        batch_size = 4
        learning_rate = 2e-4
        max_epoch = 100
        latent_size = 32
        beta = 2.0
        train_ratio = 0.9
        seed = 42
        load_epoch = -1
        base_data_path = str(Path(__file__).parent / "data")
        result_base_path = str(Path(__file__).parent / "result")
    args = DefaultArgs()

# グローバル変数でiter番号を設定
ITER = args.iter

# ハイパーパラメータ
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MAX_EPOCH = args.max_epoch
NUM_WORKERS = 0
LOAD_EPOCH = args.load_epoch

# データセット設定
GRID_SIZE = [4, 4, 4]
TRAIN_RATIO = args.train_ratio
SEED = args.seed

# 潜在変数の次元
LATENT_SIZE = args.latent_size
BETA = args.beta

# JSONファイルのパス設定（動的に生成、相対パス使用）
BASE_DATA_PATH = args.base_data_path
RESULT_BASE_PATH = args.result_base_path

STRUCTURES_DB_PATHS = [
    os.path.join(BASE_DATA_PATH, f"iter{i}_structures.json") for i in range(ITER + 1)
]

OVERPOTENTIALS_JSON_PATHS = [
    os.path.join(BASE_DATA_PATH, f"iter{i}_calculation_result.json") for i in range(ITER + 1)
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用デバイス: {device}")


# ------------------------------
# 条件付きVAE モデル定義
# ------------------------------
class ConditionalVAE(nn.Module):
    def __init__(self, latent_size=128, condition_dim=2):
        super(ConditionalVAE, self).__init__()
        self.latent_size = latent_size
        self.condition_dim = condition_dim  # 2つの条件（過電圧、Pt割合）
        self.activation = nn.LeakyReLU(0.1, inplace=False)

        # ======== エンコーダ========
        # 条件値の非線形変換（エンコーダ用）
        self.enc_label_fc1 = nn.Linear(condition_dim, 32)
        self.enc_label_fc2 = nn.Linear(32, 32)
        self.enc_label_fc3 = nn.Linear(32, 16)
        
        # 畳み込み層（入力チャンネルを16に変更：4 + 16）
        self.conv1 = nn.Conv2d(4+16, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        
        # 全結合層（μとlogvarを出力）
        self.fc1 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_size)
        self.fc_logvar = nn.Linear(512, latent_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.05)

        # ======== デコーダ========
        # 条件値の非線形変換（デコーダ用）
        self.dec_label_fc1 = nn.Linear(condition_dim, 32)
        self.dec_label_fc2 = nn.Linear(32, 32)
        self.dec_label_fc3 = nn.Linear(32, 16)

        # 全結合層（入力次元を16増やす：latent_size + 16）
        self.dec_fc1 = nn.Linear(latent_size + 16, 256)
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_fc3 = nn.Linear(512, 64 * 2 * 2)
        
        # 転置畳み込み層
        self.deconv1 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn4 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn5 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_bn6 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 12, kernel_size=3, stride=1, padding=1)

    def encode_condition_enc(self, y):
        """エンコーダ用の条件埋め込み（2つの条件対応）"""
        h1 = self.activation(self.enc_label_fc1(y))
        h2 = self.activation(self.enc_label_fc2(h1))
        h3 = self.activation(self.enc_label_fc3(h2))
        return h3

    def encode_condition_dec(self, y):
        """デコーダ用の条件埋め込み（2つの条件対応）"""
        h1 = self.activation(self.dec_label_fc1(y))
        h2 = self.activation(self.dec_label_fc2(h1))
        h3 = self.activation(self.dec_label_fc3(h2))
        return h3

    def encode(self, x, y):
        """エンコーダ：入力xと条件yから潜在変数のμとlogvarを出力"""
        batch_size = x.size(0)
        
        # 条件を非線形変換
        y_encoded = self.encode_condition_enc(y)  # [B, 16]
        
        # 条件を空間的に拡張
        y_expanded = y_encoded.view(batch_size, -1, 1, 1).expand(-1, -1, 8, 8)  # [B, 16, 8, 8]
        
        # 入力と条件を結合
        x_cond = torch.cat([x, y_expanded], dim=1)  # [B, 20, 8, 8]
        
        # 畳み込み層
        h = self.activation(self.conv1(x_cond))  # [B, 256, 8, 8]
        h = self.dropout(h)
        
        h = self.activation(self.conv2(h))  # [B, 512, 4, 4]
        h = self.dropout(h)
        
        h = self.activation(self.conv3(h))  # [B, 1024, 2, 2]
        
        # Global Average Pooling
        h = F.adaptive_avg_pool2d(h, (1, 1)).view(batch_size, -1)  # [B, 1024]
        
        # 全結合層
        h = self.activation(self.fc1(h))  # [B, 512]
        h = self.dropout(h)
        
        mu = self.fc_mu(h)      # [B, latent_size]
        logvar = self.fc_logvar(h)  # [B, latent_size]
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        """デコーダ：潜在変数zと条件yから再構成を出力"""
        # 条件を非線形変換
        y_encoded = self.encode_condition_dec(y)  # [B, 16]
        
        # 潜在変数と条件を結合
        z_cat = torch.cat((z, y_encoded), dim=1)  # [B, latent_size + 16]
        
        # 全結合層
        h = self.activation(self.dec_fc1(z_cat))
        h = self.activation(self.dec_fc2(h))
        h = self.activation(self.dec_fc3(h))
        
        # reshape
        h = h.view(h.size(0), 64, 2, 2)
        
        # 逆畳み込み層
        h = self.activation(self.dec_bn4(self.deconv1(h)))  # [B, 128, 4, 4]
        h = self.activation(self.dec_bn5(self.deconv2(h)))  # [B, 64, 8, 8]
        h = self.activation(self.dec_bn6(self.deconv3(h)))  # [B, 32, 8, 8]
        output = self.deconv4(h)  # [B, 12, 8, 8]
        
        return output

    def forward(self, x, y):
        """順伝播"""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

# ------------------------------
# 損失関数
# ------------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1):
    """
    VAE損失関数（スケール調整済み）
    recon_x: [B, 12, 8, 8] - デコーダ出力（logits）
    x: [B, 4, 8, 8] - 正解ラベル（0,1,2のクラスインデックス）
    """
    x = x.to(dtype=torch.long)
    
    class_weights = torch.tensor([0.1, 1.0, 1.0], device=x.device)
    
    # 各層の損失を計算
    recon_loss = 0
    for z in range(4):
        layer_pred = recon_x[:, z*3:(z+1)*3]  # [B, 3, 8, 8]
        layer_target = x[:, z]  # [B, 8, 8]
        
        recon_loss += F.cross_entropy(
            layer_pred, 
            layer_target, 
            weight=class_weights,
            reduction='sum' 
        )
    
    # KL損失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# ------------------------------
# 学習・評価関数
# ------------------------------
def train_vae(epoch, model, train_loader, optimizer, beta):
    model.train()
    train_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    start_time = time.time()
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        try:
            data = data.to(device).float()
            # 2つの条件ラベルに対応（[B, 2]の形状を維持）
            labels = labels.to(device).float()  # [B, 2]
            
            optimizer.zero_grad()
            
            recon, mu, logvar = model(data, labels)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")
                
        except Exception as e:
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_total / len(train_loader.dataset)
    avg_kl = kl_loss_total / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl, elapsed

def test_vae(epoch, model, test_loader, beta):
    model.eval()
    test_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, labels in test_loader:
            try:
                data = data.to(device).float()
                # 2つの条件ラベルに対応（[B, 2]の形状を維持）
                labels = labels.to(device).float()  # [B, 2]
                
                recon, mu, logvar = model(data, labels)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
                
                test_loss += loss.item()
                recon_loss_total += recon_loss.item()
                kl_loss_total += kl_loss.item()
                
            except Exception as e:
                traceback.print_exc()
                continue
    
    elapsed = time.time() - start_time
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon = recon_loss_total / len(test_loader.dataset)
    avg_kl = kl_loss_total / len(test_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl, elapsed

# ------------------------------
# 学習曲線のプロット
# ------------------------------
def plot_learning_curves(train_losses, test_losses, result_dir):
    """学習曲線をプロット"""
    epochs = np.arange(1, len(train_losses) + 1)
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses[:, 0], 'b-', label='Train Total Loss')
    plt.plot(epochs, test_losses[:, 0], 'r-', label='Test Total Loss')
    plt.title('Total Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_losses[:, 1], 'b-', label='Train Recon Loss')
    plt.plot(epochs, test_losses[:, 1], 'r-', label='Test Recon Loss')
    plt.title('Reconstruction Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_losses[:, 2], 'b-', label='Train KL Loss')
    plt.plot(epochs, test_losses[:, 2], 'r-', label='Test KL Loss')
    plt.title('KL Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{result_dir}/learning_curves.png")
    plt.close()

# ------------------------------
# メイン処理
# ------------------------------
def main():
    # コマンドライン引数の表示
    print(f"=== Conditional VAE 学習 設定 ===")
    print(f"iter: {ITER}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"max_epoch: {MAX_EPOCH}")
    print(f"latent_size: {LATENT_SIZE}")
    print(f"beta: {BETA}")
    print(f"train_ratio: {TRAIN_RATIO}")
    print(f"seed: {SEED}")
    print(f"load_epoch: {LOAD_EPOCH}")
    print(f"base_data_path: {BASE_DATA_PATH}")
    print(f"result_base_path: {RESULT_BASE_PATH}")
    print("=" * 40)
    
    # 結果保存先を動的に設定（相対パス使用）
    result_dir = os.path.join(RESULT_BASE_PATH, f"iter{ITER}")
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"=== Conditional VAE 学習 (iter{ITER}) ===")
    print(f"使用するデータ: iter0 ~ iter{ITER}")
    print(f"構造DB: {len(STRUCTURES_DB_PATHS)}個のファイル")
    print(f"過電圧JSON: {len(OVERPOTENTIALS_JSON_PATHS)}個のファイル")
    print(f"結果保存先: {result_dir}")
    
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch = MAX_EPOCH
    num_workers = NUM_WORKERS
    load_epoch = LOAD_EPOCH
    latent_size = LATENT_SIZE
    beta = BETA

    # モデル作成前にも再度クリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # データローダーの作成
    train_loader, test_loader, dataset = make_data_loaders_from_json(
        structures_db_paths=STRUCTURES_DB_PATHS,
        overpotentials_json_paths=OVERPOTENTIALS_JSON_PATHS,
        use_binary_labels=True,
        train_ratio=TRAIN_RATIO,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=SEED,
        grid_size=GRID_SIZE
    )
    
    print(f"データセットのサイズ: {len(dataset)}")
    print(f"訓練データ数: {len(train_loader.dataset)}, テストデータ数: {len(test_loader.dataset)}")
    
    # モデルの初期化（2つの条件ラベルに対応）
    model = ConditionalVAE(latent_size=latent_size, condition_dim=2).to(device)
    print("Conditional VAE（2つの条件ラベル対応）を作成しました")
    
    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 損失履歴を保存するリスト
    train_loss_list = []
    test_loss_list = []
    
    # 学習ループ
    for epoch in range(load_epoch + 1, max_epoch):
        print(f"\nエポック {epoch+1}/{max_epoch} 開始")
        
        train_loss, train_recon, train_kl, train_time = train_vae(
            epoch, model, train_loader, optimizer, beta
        )
        test_loss, test_recon, test_kl, test_time = test_vae(
            epoch, model, test_loader, beta
        )
        
        print(f"Epoch: {epoch+1}/{max_epoch}")
        print(f"Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Test  - Total: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
        print(f"訓練時間: {train_time:.2f}秒, 評価時間: {test_time:.2f}秒")
        
        # 損失履歴を追加
        train_loss_list.append([train_loss, train_recon, train_kl])
        test_loss_list.append([test_loss, test_recon, test_kl])
    
    # 最終結果の保存
    train_loss_array = np.array(train_loss_list)
    test_loss_array = np.array(test_loss_list)
    
    # 損失値をNumPy配列として保存
    np.save(f"{result_dir}/train_loss.npy", train_loss_array)
    np.save(f"{result_dir}/test_loss.npy", test_loss_array)
    
    # 学習曲線をプロット・保存
    plot_learning_curves(train_loss_array, test_loss_array, result_dir)
    
    # 最終モデルの保存（動的にファイル名を生成）
    torch.save(model.state_dict(), f"{result_dir}/final_cvae_iter{ITER}.pt")
    
    print(f"学習結果は {result_dir} に保存されました")
    print(f"保存されたモデル: final_cvae_iter{ITER}.pt")

if __name__ == "__main__":
    freeze_support()
    main()