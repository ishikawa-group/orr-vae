import os
import sys
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.multiprocessing import freeze_support

from tool import make_data_loaders_from_json

# ------------------------------
# ハイパーパラメータの設定
# ------------------------------
batch_size    = 4
learning_rate = 2e-4
max_epoch     = 50
num_workers   = 0  # マルチプロセス回避のために0に設定
load_epoch    = -1     # ロードするエポック（-1なら新規学習）

# データセット設定
GRID_SIZE = [4, 4, 4]  # Pt-Pdスラブの形状
TRAIN_RATIO = 0.9
SEED = 42

# 潜在変数の次元
latent_size   = 128

# JSONファイルのパス設定
STRUCTURES_DB_PATHS = [
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data/iter0_structures.json",
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data/iter1_structures.json",
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data/iter2_structures.json",
]

# 過電圧JSONファイルも複数指定
OVERPOTENTIALS_JSON_PATHS = [
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data/iter0_calculation_result.json",
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data/iter1_calculation_result.json",
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/data/iter2_calculation_result.json",
]


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用デバイス: {device}")

# ------------------------------
# モデル定義（条件付きGAN）
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_size=128, condition_dim=1):
        """
        潜在変数と条件から構造を生成するジェネレーター
        latent_size: 潜在変数の次元数
        condition_dim: 条件の次元数
        """
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.condition_dim = condition_dim
        self.activation = nn.LeakyReLU(0.1, inplace=False)

        # 条件値の非線形変換
        self.label_fc1 = nn.Linear(condition_dim, 16)
        self.label_fc2 = nn.Linear(16, 16)
        self.label_fc3 = nn.Linear(16, 8)

        # ノイズと条件の結合後の処理
        self.fc1 = nn.Linear(latent_size + 8, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 64 * 2 * 2)
        self.bn3 = nn.BatchNorm1d(64 * 2 * 2)
        
        # 畳み込み層でサイズを拡大
        self.deconv1 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.sa_block_1 = Self_Attention(32)
        self.deconv4 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=1, padding=1)

    def encode_condition(self, y):
        """条件を非線形変換"""
        h1 = self.activation(self.label_fc1(y))
        h2 = self.activation(self.label_fc2(h1))
        h3 = self.activation(self.label_fc3(h2))
        return h3

    def forward(self, z, y):
        """
        z: ランダムノイズベクトル [B, latent_size]
        y: 条件値
        """
        # 条件を非線形変換
        y_encoded = self.encode_condition(y)
        
        # ノイズと条件を結合
        z_cat = torch.cat((z, y_encoded), dim=1)
        
        # 全結合層を通して特徴マップに変換
        h = self.activation(self.bn1(self.fc1(z_cat)))
        h = self.activation(self.bn2(self.fc2(h)))
        h = self.activation(self.bn3(self.fc3(h)))
        
        # [B, 64*2*2] -> [B, 64, 2, 2]に変形
        h = h.view(h.size(0), 64, 2, 2)
        
        # 逆畳み込み層でサイズを拡大
        h = self.activation(self.bn4(self.deconv1(h)))  # -> [B, 128, 4, 4]
        h = self.activation(self.bn5(self.deconv2(h)))  # -> [B, 64, 8, 8]
        h = self.activation(self.bn6(self.deconv3(h)))  # -> [B, 32, 8, 8]
        h = self.sa_block_1(h)  # Self-Attentionを適用 [B, 32, 8, 8]
        # 最終出力 [B, 4, 8, 8]
        output = self.deconv4(h)
        
        return output


class Discriminator(nn.Module):
    def __init__(self, condition_dim=1):
        """
        構造と条件から真偽を判定するディスクリミネーター
        """
        super(Discriminator, self).__init__()
        self.condition_dim = condition_dim
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.dropout = nn.Dropout(0.05)
        
        # 条件値の非線形変換
        self.label_fc1 = nn.Linear(condition_dim, 8)
        self.label_fc2 = nn.Linear(8, 8)
        self.label_fc3 = nn.Linear(8, 4)
        
        # 畳み込み層（カーネルサイズ3のみ使用、BatchNormなし）
        # [B, 4+4, 8, 8] -> [B, 64, 8, 8]
        self.conv1 = spectral_norm(nn.Conv2d(4+4, 256, kernel_size=3, stride=1, padding=1))
        self.sa_block_2 = Self_Attention(256)  # Self-Attentionを適用
        
        # [B, 64, 8, 8] -> [B, 128, 4, 4]
        self.conv2 = spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1))
        
        # [B, 128, 4, 4] -> [B, 256, 2, 2]
        self.conv3 = spectral_norm(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1))
        
        # 全結合層
        # Global Average Pooling後の特徴数は1024
        self.fc1 = spectral_norm(nn.Linear(1024, 512))
        self.fc2 = spectral_norm(nn.Linear(512, 1))  # 出力は単一スカラー
        
    def encode_condition(self, y):
        """条件を非線形変換"""
        h1 = self.activation(self.label_fc1(y))
        h2 = self.activation(self.label_fc2(h1))
        h3 = self.activation(self.label_fc3(h2))
        return h3
        
    def forward(self, x, y):
        """
        x: スラブ構造 [B, 4, 8, 8]
        y: 条件値
        """
        batch_size = x.size(0)
        
        # 条件を非線形変換
        y_encoded = self.encode_condition(y)  # [B, 4]
        
        # 条件を空間的に拡張してチャネルとして追加
        y_expanded = y_encoded.view(batch_size, -1, 1, 1).expand(-1, -1, 8, 8)  # [B, 4, 8, 8]
        
        # 入力と条件を結合
        x_cond = torch.cat([x, y_expanded], dim=1)  # [B, 4+4, 8, 8]
        
        # 畳み込み層（スペクトル正規化済み、BatchNormなし）
        h = self.activation(self.conv1(x_cond))  # [B, 64, 8, 8]
        h = self.sa_block_2(h)  # Self-Attentionを適用 [B, 256, 8, 8]
        h = self.dropout(h)
        
        h = self.activation(self.conv2(h))  # [B, 128, 4, 4]
        h = self.dropout(h)
        
        h = self.activation(self.conv3(h))  # [B, 256, 2, 2]
        
        # Global Average Pooling
        h = F.adaptive_avg_pool2d(h, (1, 1)).view(batch_size, -1)  # [B, 256]
        
        # 全結合層
        h = self.activation(self.fc1(h))  # [B, 256]
        h = self.dropout(h)
        h = self.fc2(h)  # [B, 1]
        
        return h

# ------------------------------
# 学習用のユーティリティ関数
# ------------------------------

## Self_Attention Block の定義
class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        dk = max(1, in_dim // 4)  # 中間チャネル数 (d_k)。0 になるケースを防ぐ
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=dk, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=dk, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  # 行方向 (Query ごと) 正規化
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(dk)  # √d_k でのスケーリング

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W  # 空間トークン数
        q = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # B, N, d_k
        k = self.key_conv(x).view(B, -1, N)  # B, d_k, N
        v = self.value_conv(x).view(B, C, N)  # B, C, N
        S = torch.bmm(q, k) / self.scale  # B, N, N (√d_k で割る)
        attn = self.softmax(S)  # Softmax(dim=-1)
        o = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)  # B, C, H, W
        out = x + self.gamma * o  # 残差 + 学習可係数
        return out

# ------------------------------
# 損失関数
# ------------------------------
def generator_loss(d_fake_output):
    """ジェネレーターの損失関数"""
    targets = torch.ones_like(d_fake_output).to(device)
    loss = F.binary_cross_entropy_with_logits(d_fake_output, targets)
    return loss

def discriminator_loss(d_real_output, d_fake_output):
    """ディスクリミネーターの損失関数"""
    # 本物データを本物と判定する損失
    real_targets = torch.ones_like(d_real_output).to(device)    
    real_loss = F.binary_cross_entropy_with_logits(d_real_output, real_targets)
    
    # 偽データを偽物と判定する損失
    fake_targets = torch.zeros_like(d_fake_output).to(device)
    fake_loss = F.binary_cross_entropy_with_logits(d_fake_output, fake_targets)
    
    return real_loss + fake_loss


# ------------------------------
# 学習ループ
# ------------------------------
def train_gan(epoch, generator, discriminator, train_loader, g_optimizer, d_optimizer):
    generator.train()
    discriminator.train()
    
    g_loss_total = 0
    d_loss_total = 0
    start_time = time.time()
    
    for i, (real_data, real_labels) in enumerate(train_loader):
        try:
            batch_size = real_data.size(0)
            real_data = real_data.to(device).float()
            real_labels = real_labels.to(device).view(-1, 1).float()
            
            # ---------------------
            # Discriminatorの学習
            # ---------------------
            d_optimizer.zero_grad()
            
            # 本物データでの損失
            d_real = discriminator(real_data, real_labels)
            
            # 偽データの生成
            z = torch.randn(batch_size, generator.latent_size).to(device)
            fake_data = generator(z, real_labels)
            
            # 偽データでの損失
            d_fake = discriminator(fake_data.detach(), real_labels)
            
            # 修正: 定義済みの損失関数を使用
            d_loss = discriminator_loss(d_real, d_fake)
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            # Generatorの学習
            # ---------------------
            g_optimizer.zero_grad()
            
            # Generatorの損失（Discriminatorを騙すことが目標）
            d_fake = discriminator(fake_data, real_labels)
            
            # 修正: 定義済みの損失関数を使用
            g_loss = generator_loss(d_fake)
            
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            
            if i % 1 == 0:
                print(f"Batch {i}/{len(train_loader)}: G_Loss={g_loss.item():.4f}, D_Loss={d_loss.item():.4f}")
                
        except Exception as e:
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    return g_loss_total / len(train_loader), d_loss_total / len(train_loader), elapsed

def test_gan(epoch, generator, discriminator, test_loader, result_dir):
    generator.eval()
    discriminator.eval()
    
    g_loss_total = 0
    d_loss_total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (real_data, real_labels) in enumerate(test_loader):
            try:
                batch_size = real_data.size(0)
                real_data = real_data.to(device).float()
                real_labels = real_labels.to(device).view(-1, 1).float()
                
                # Discriminatorの評価
                d_real = discriminator(real_data, real_labels)
                
                z = torch.randn(batch_size, generator.latent_size).to(device)
                fake_data = generator(z, real_labels)
                
                d_fake = discriminator(fake_data, real_labels)
                
                # 修正: 定義済みの損失関数を使用
                d_loss = discriminator_loss(d_real, d_fake)
                g_loss = generator_loss(d_fake)
                
                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()
                    
            except Exception as e:
                traceback.print_exc()
                continue
    
    elapsed = time.time() - start_time
    return g_loss_total / len(test_loader), d_loss_total / len(test_loader), elapsed


# ------------------------------
# 学習曲線のプロット
# ------------------------------
def plot_learning_curves(train_loss, test_loss, result_dir):
    """学習曲線をプロット"""
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss[:, 0], 'b-', label='Train Generator Loss')
    plt.plot(epochs, train_loss[:, 1], 'r-', label='Train Discriminator Loss')
    plt.title('Train Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_loss[:, 0], 'b-', label='Test Generator Loss')
    plt.plot(epochs, test_loss[:, 1], 'r-', label='Test Discriminator Loss')
    plt.title('Test Loss over Epochs', fontsize=14)
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
    # 結果保存先を設定
    result_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccgan/result/iter2"
    os.makedirs(result_dir, exist_ok=True)
    
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
    
    # モデルの初期化
    generator = Generator(latent_size=latent_size, condition_dim=1).to(device)
    discriminator = Discriminator(condition_dim=1).to(device)
    print("Generator と Discriminator を作成しました")
    
    # オプティマイザの設定
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # 損失履歴を保存するリスト
    train_loss_list = []
    test_loss_list = []
    
    # 学習ループ
    for epoch in range(load_epoch + 1, max_epoch):
        print(f"\nエポック {epoch+1}/{max_epoch} 開始")
        
        train_g_loss, train_d_loss, train_time = train_gan(
            epoch, generator, discriminator, train_loader, g_optimizer, d_optimizer
        )
        test_g_loss, test_d_loss, test_time = test_gan(
            epoch, generator, discriminator, test_loader, result_dir
        )
        
        print(f"Epoch: {epoch+1}/{max_epoch}")
        print(f"Train - G_Loss: {train_g_loss:.4f}, D_Loss: {train_d_loss:.4f}")
        print(f"Test  - G_Loss: {test_g_loss:.4f}, D_Loss: {test_d_loss:.4f}")
        print(f"訓練時間: {train_time:.2f}秒, 評価時間: {test_time:.2f}秒")
        
        # 損失履歴を追加
        train_loss_list.append([train_g_loss, train_d_loss])
        test_loss_list.append([test_g_loss, test_d_loss])
    
    # 最終結果の保存
    train_loss_array = np.array(train_loss_list)
    test_loss_array = np.array(test_loss_list)
    
    # 損失値をNumPy配列として保存
    np.save(f"{result_dir}/train_loss.npy", train_loss_array)
    np.save(f"{result_dir}/test_loss.npy", test_loss_array)
    
    # 学習曲線をプロット・保存
    plot_learning_curves(train_loss_array, test_loss_array, result_dir)
    
    # 最終モデルの保存
    torch.save(generator.state_dict(), f"{result_dir}/final_generator_iter2.pt")
    torch.save(discriminator.state_dict(), f"{result_dir}/final_discriminator_iter2.pt")
    
    
    print(f"学習結果は {result_dir} に保存されました")

if __name__ == "__main__":
    freeze_support()
    main()