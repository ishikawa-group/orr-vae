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
from torch.multiprocessing import freeze_support

from data_loader_from_json import make_data_loaders_from_json

# ------------------------------
# ハイパーパラメータの設定
# ------------------------------
batch_size    = 32
learning_rate = 1e-3
max_epoch     = 40
num_workers   = 0  # マルチプロセス回避のために0に設定
load_epoch    = -1     # ロードするエポック（-1なら新規学習）

# データセット設定
GRID_SIZE = [4, 4, 4]  # Pt-Pdスラブの形状
TRAIN_RATIO = 0.8
SEED = 42

# 潜在変数の次元
latent_size   = 128

# JSONファイルのパス設定
STRUCTURES_DB_PATHS = [
    "./data/iter0_structure.json",
]

# 過電圧JSONファイルも複数指定
OVERPOTENTIALS_JSON_PATHS = [
    "./results/iter0_overpotentials.json",
]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用デバイス: {device}")

# ------------------------------
# モデル定義（条件付きVAE - 連続値対応版）
# ------------------------------
class CVAE(nn.Module):
    def __init__(self, latent_size=64, condition_dim=1):
        """
        latent_size : 潜在変数の次元数
        condition_dim : 条件の次元数（Pd比率の連続値は1次元）
        """
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.condition_dim = condition_dim
        self.activation = nn.LeakyReLU(0.1, inplace=False)

        # 条件値の非線形変換（エンコーダ用）
        self.enc_label_fc1 = nn.Linear(condition_dim, 32)
        self.enc_label_fc2 = nn.Linear(32, 32)
        self.enc_label_fc3 = nn.Linear(32, 16)

        # Encoder - 入力は結合後のテンソル [B, 12, 8, 8]
        # 4(入力チャネル) + 16(条件埋め込み) = 20チャネル
        self.conv1   = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.linear1 = nn.Linear(128 * 2 * 2, 256)
        self.mu      = nn.Linear(256, self.latent_size)
        self.logvar  = nn.Linear(256, self.latent_size)

        # 条件値の非線形変換(デコーダ用)
        self.label_fc1 = nn.Linear(condition_dim, 32)
        self.label_fc2 = nn.Linear(32, 32)
        self.label_fc3 = nn.Linear(32, 16)

        # Decoder
        self.linear2 = nn.Linear(self.latent_size + 16, 256)
        self.linear3 = nn.Linear(256, 64 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn6     = nn.BatchNorm2d(16)
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=4*3, kernel_size=3, stride=1, padding=1)

    def encoder_label(self, y):
        """
        エンコーダ用の条件埋め込み生成
        y : [B, 1] (Pd比率の値)
        """
        h1 = self.activation(self.enc_label_fc1(y))
        h2 = self.activation(self.enc_label_fc2(h1))
        h3 = self.activation(self.enc_label_fc3(h2))

        return h3
    
    def encoder(self, x, label_embedding):
        """
        x : 入力スラブテンソル [B, 4, 8, 8]
        label_embedding : 条件埋め込みテンソル [B, 8]
        """
        # 条件埋め込みをチャネルとして拡張
        batch_size = x.size(0)
        label_channels = label_embedding.size(1)
        
        # 条件埋め込みを各位置に拡張 [B, 8] -> [B, 8, 8, 8]
        label_expanded = label_embedding.view(batch_size, label_channels, 1, 1)
        label_expanded = label_expanded.expand(-1, -1, x.size(2), x.size(3))
        
        # チャネル方向に結合 [B, 4, 8, 8] + [B, 8, 8, 8] -> [B, 12, 8, 8]
        t = torch.cat((x, label_expanded), dim=1)
        
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
        t = self.activation(self.bn5(self.deconv3(t)))
        t = self.deconv4(t)  

        return t

    def label_encoder(self, y):
        """
        連続値を非線形変換
        y : [B, 1] (Pd比率の値)
        """
        h1 = self.activation(self.label_fc1(y))
        h2 = self.activation(self.label_fc2(h1))
        h3 = self.activation(self.label_fc3(h2))

        return h3

    def forward(self, x, y):
        """
        x : 入力スラブ [B, 4, 8, 8]
        y : Pd比率値 [B, 1]
        """
        # 条件を非線形変換（エンコーダ用とデコーダ用で別々に処理）
        enc_label_embedding = self.encoder_label(y)
        dec_label_embedding = self.label_encoder(y)
        
        # エンコーダで条件付き変換
        mu, logvar = self.encoder(x, enc_label_embedding)
        z = self.reparameterize(mu, logvar)
        
        # デコーダで条件付き生成
        z_cat = torch.cat((z, dec_label_embedding), dim=1)
        pred = self.decoder(z_cat)
        
        return pred, mu, logvar

# ------------------------------
# 損失関数
# ------------------------------
def loss_function(x, pred, mu, logvar):
    # pred: [B, 12, H, W] - logits
    # x: [B, 4, H, W] - クラスインデックス（各層ごとに0,1,2のいずれか）
    
    x = x.to(dtype=torch.long)
    
    # クラスごとの重みを設定（0=空:倍、1=Pd:10倍、2=Pt:10倍）
    class_weights = torch.tensor([1.0, 10.0, 10.0], device=x.device)
    
    # 各層ごとに処理する場合
    recon_loss = 0
    for z in range(4):  # 4層
        # 各層で3クラス分類
        layer_pred = pred[:, z*3:(z+1)*3]  # 各層に3チャネル
        recon_loss += F.cross_entropy(
            layer_pred, 
            x[:, z], 
            weight=class_weights,
            reduction='sum'
        )
    
    # KL損失
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld

# ------------------------------
# 可視化関数
# ------------------------------
def visualize_slabs(pred, labels, result_dir):
    """Visualize the reconstructed structures with gridlines."""
    batch_size = min(8, pred.size(0))  # Visualize up to 8 structures
    
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
    
    for i in range(batch_size):
        for z in range(4):
            # Get the softmax of the predicted results
            layer_pred = pred[i, z * 3:(z + 1) * 3]
            layer_pred = F.softmax(layer_pred, dim=0)
            
            # Get the class with the highest probability
            _, predicted_class = torch.max(layer_pred, dim=0)
            
            # Set the colormap (0: white, 1: blue, 2: red)
            cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            # Visualize the predicted class with gridlines
            im = axes[i, z].imshow(predicted_class.numpy(), cmap=cmap, norm=norm)
            axes[i, z].set_title(f'Layer {z}')
            axes[i, z].axis('on')
            axes[i, z].grid(color='black', linestyle='--', linewidth=0.5)
            axes[i, z].set_xticks(np.arange(-0.5, predicted_class.size(1), 1), minor=True)
            axes[i, z].set_yticks(np.arange(-0.5, predicted_class.size(0), 1), minor=True)
            axes[i, z].tick_params(which='minor', length=0)  # Hide minor ticks
        
        # Display the Pd ratio
        axes[i, 4].text(0.5, 0.5, f'Pd Ratio: {labels[i].item():.3f}', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=14)
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/reconstructed_samples.png")
    plt.close()

def generate_samples(model, result_dir, num_samples=8):
    """新しいサンプルを生成"""
    model.eval()
    with torch.no_grad():
        # ランダムな潜在変数
        z = torch.randn(num_samples, model.latent_size).to(device)
        
        # 異なるPd比率の条件を設定
        pd_ratios = torch.linspace(0.125, 1, num_samples).view(-1, 1).to(device)
        
        # 条件を非線形変換
        label_embedding = model.label_encoder(pd_ratios)
        
        # 潜在変数と条件を結合してデコード
        z_cat = torch.cat((z, label_embedding), dim=1)
        pred = model.decoder(z_cat)
        
        # 生成された構造の可視化
        visualize_slabs(pred.cpu(), pd_ratios.cpu(), result_dir)
        
        print(f"条件付き生成サンプルを {result_dir}/generated_samples.png に保存しました")

# ------------------------------
# 学習ループ
# ------------------------------
def train(epoch, model, train_loader, optimizer, kld_weight=0.5):
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
            
            # エポックが進むにつれてKL項の重みを徐々に増やす
            beta = min(1.0, epoch / (max_epoch)) * kld_weight
            #beta = 1
            loss = recon_loss + beta * kld
            
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kld   += kld.item()
            
            if i % 10 == 0:
                print(f"Batch {i}/{len(train_loader)}: Loss={loss.item():.4f}")
                
        except Exception as e:
            traceback.print_exc()
            continue
    
    dataset_size = len(train_loader.dataset)
    elapsed = time.time() - start_time
    return total_loss / len(train_loader), total_kld / len(train_loader), total_recon / len(train_loader), elapsed

def test(epoch, model, test_loader, result_dir, kld_weight=0.1):
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
                
                # 訓練と同じベータ値を使用
                beta = min(1.0, epoch / (max_epoch * 0.8)) * kld_weight
                loss = recon_loss + beta * kld

                total_loss  += loss.item()
                total_recon += recon_loss.item()
                total_kld   += kld.item()
                
                # 最終エポックのみ可視化
                if epoch == max_epoch - 1 and i == 0:
                    visualize_slabs(pred.cpu(), y.cpu(), result_dir)
                    
            except Exception as e:
                traceback.print_exc()
                continue

    elapsed = time.time() - start_time
    return total_loss / len(test_loader), total_kld / len(test_loader), total_recon / len(test_loader), elapsed

# ------------------------------
# 学習曲線のプロット
# -----------------------------
def plot_learning_curves(train_loss, test_loss, result_dir):
    """Plot learning curves and save the final figure."""
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss[:, 0], 'b-', label='Train Total Loss')
    plt.plot(epochs, test_loss[:, 0], 'r-', label='Test Total Loss')
    plt.title('Total Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_loss[:, 1], 'b-', label='Train KL Loss')
    plt.plot(epochs, test_loss[:, 1], 'r-', label='Test KL Loss')
    plt.title('KL Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_loss[:, 2], 'b-', label='Train Reconstruction Loss')
    plt.plot(epochs, test_loss[:, 2], 'r-', label='Test Reconstruction Loss')
    plt.title('Reconstruction Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
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
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 修正: 新しいデータローダーを使用
    train_loader, test_loader, dataset = make_data_loaders_from_json(
        structures_db_paths=STRUCTURES_DB_PATHS,
        overpotentials_json_paths=OVERPOTENTIALS_JSON_PATHS,
        train_ratio=TRAIN_RATIO, 
        batch_size=batch_size,
        num_workers=num_workers,
        seed=SEED,
        grid_size=GRID_SIZE
    )
    
    print(f"データセットのサイズ: {len(dataset)}")
    print(f"訓練データ数: {len(train_loader.dataset)}, テストデータ数: {len(test_loader.dataset)}")
    print(f"過電圧の範囲: {dataset.target_min:.3f} 〜 {dataset.target_max:.3f}")
    
    # モデルの初期化
    model = CVAE(latent_size=latent_size, condition_dim=1).to(device)
    print("モデルを作成しました")
    
    # 既存モデルのロード（必要な場合）
    if load_epoch > 0:
        checkpoint_path = f'{result_dir}/model_{load_epoch}.pt'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"モデル {load_epoch} をロードしました")
    
    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 損失履歴を保存するリスト
    train_loss_list = []
    test_loss_list = []
    
    # 学習ループ
    for epoch in range(load_epoch + 1, max_epoch):
        print(f"\nエポック {epoch+1}/{max_epoch} 開始")
        
        train_total, train_kld, train_recon, train_time = train(epoch, model, train_loader, optimizer)
        test_total, test_kld, test_recon, test_time = test(epoch, model, test_loader, result_dir)
        
        print(f"Epoch: {epoch+1}/{max_epoch} | 訓練損失: {train_total:.4f} | テスト損失: {test_total:.4f}")
        print(f"訓練時間: {train_time:.2f}秒, 評価時間: {test_time:.2f}秒")
        
        # 損失履歴を追加
        train_loss_list.append([train_total, train_kld, train_recon])
        test_loss_list.append([test_total, test_kld, test_recon])
        
        # 5エポックごとにチェックポイントを保存
        #if (epoch + 1) % 5 == 0 or epoch == max_epoch - 1:
        #    torch.save(model.state_dict(), f"{result_dir}/model_{epoch+1}.pt")
        #    print(f"モデルをエポック {epoch+1} で保存しました")
    
    # 最終結果の保存
    train_loss_array = np.array(train_loss_list)
    test_loss_array = np.array(test_loss_list)
    
    # 損失値をNumPy配列として保存
    np.save(f"{result_dir}/train_loss.npy", train_loss_array)
    np.save(f"{result_dir}/test_loss.npy", test_loss_array)
    
    # 学習曲線をプロット・保存
    plot_learning_curves(train_loss_array, test_loss_array, result_dir)
    
    # 最終モデルの保存
    torch.save(model.state_dict(), f"{result_dir}/final_model.pt")
    
    # 生成サンプルを作成
    generate_samples(model, result_dir)
    
    print(f"学習結果は {result_dir} に保存されました")

if __name__ == "__main__":
    freeze_support()  # Windows環境でのマルチプロセス対応
    main()