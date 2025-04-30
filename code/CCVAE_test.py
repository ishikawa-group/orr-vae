import os
import sys
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

# ------------------------------
# ハイパーパラメータの設定
# ------------------------------
batch_size    = 256
learning_rate = 1e-3
max_epoch     = 50
num_workers   = 4
load_epoch    = -1     # ロードするエポック（-1なら新規学習）
generate      = True

# 潜在変数の次元
latent_size   = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ------------------------------
# モデル定義（条件付きVAE）
# ------------------------------
class CVAE(nn.Module):
    def __init__(self, latent_size=32, num_classes=10):
        """
        latent_size : 潜在変数の次元数
        num_classes : 分類クラス数（MNISTの場合は10）
        """
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.activation = nn.SiLU()  # 勾配が安定しやすい活性化関数

        # Encoder
        # 入力は画像 x (1チャンネル) と条件ラベル（後で1チャンネルとして結合）
        self.conv1   = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=2)
        self.bn1     = nn.BatchNorm2d(16)
        self.conv2   = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2     = nn.BatchNorm2d(32)
        self.linear1 = nn.Linear(4 * 4 * 32, 300)
        self.mu      = nn.Linear(300, self.latent_size)
        self.logvar  = nn.Linear(300, self.latent_size)

        # ラベルの非線形変換（スカラー値を32次元に変換）
        self.label_fc1 = nn.Linear(1, 64)
        self.label_fc2 = nn.Linear(64,16)

        # Decoder
        # 入力は潜在変数 (latent_size) とラベル埋め込み (32) の結合＝ latent_size+16
        self.linear2 = nn.Linear(self.latent_size + 16, 300)
        self.linear3 = nn.Linear(300, 4 * 4 * 32)
        self.conv3   = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2)
        self.bn3     = nn.BatchNorm2d(16)
        self.conv4   = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2)
        self.bn4     = nn.BatchNorm2d(1)
        # 最終出力が28×28になるようにkernel_sizeのみ指定（strideは1がデフォルト）
        self.conv5   = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4)

    def encoder(self, x, y):
        """
        エンコーダ部  
        x : 入力画像テンソル [B, 1, 28, 28]  
        y : ワンホット化されたラベル [B, num_classes]
        """
        # ラベルはワンホットベクトルからカテゴリ番号を取得し，
        # その値を各画像領域に広げて結合（画像サイズと同じ形状に拡大）
        y_label = torch.argmax(y, dim=1).view(-1, 1, 1, 1)
        y_expanded = torch.ones_like(x) * y_label.to(x.device)
        # 入力チャネルを2にするため画像とラベルを結合
        t = torch.cat((x, y_expanded), dim=1)
        
        t = self.activation(self.bn1(self.conv1(t)))
        t = self.activation(self.bn2(self.conv2(t)))
        t = t.view(t.size(0), -1)
        t = self.activation(self.linear1(t))
        
        mu     = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        再パラメータ化トリック
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return eps * std + mu

    def unflatten(self, x):
        """
        1次元のテンソルを畳み込み層用に整形 [B, 32, 4, 4] に変換
        ※ ここで出力チャネルは固定（32）としています。
        """
        return x.view(x.size(0), 32, 4, 4)

    def decoder(self, z):
        """
        デコーダ部  
        z : 潜在変数とラベル埋め込みの結合 [B, latent_size+16]
        """
        t = self.activation(self.linear2(z))
        t = self.activation(self.linear3(t))
        t = self.unflatten(t)
        t = self.activation(self.bn3(self.conv3(t)))
        t = self.activation(self.bn4(self.conv4(t)))
        # 最終層は出力値を整えるためにそのまま出力（必要に応じてSigmoid等を利用）
        t = self.activation(self.conv5(t))
        return t

    def label_encoder(self, y):
        """
        ラベルを2層のニューラルネットで非線形変換する  
        y : [B, 1]  (0～9の値そのまま)
        出力 : [B, 32]
        """
        h1 = self.activation(self.label_fc1(y))
        h2 = self.activation(self.label_fc2(h1))
        return h2

    def forward(self, x, y):
        """
        x : 入力画像 [B,1,28,28]
        y : ワンホットラベル [B,num_classes]
        """
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        # ラベルはそのままの値（0～9）を使い、非線形変換により16要素のベクトルに変換
        digit_labels = torch.argmax(y.float(), dim=1).view(-1, 1).float()
        label_embedding = self.label_encoder(digit_labels.to(x.device))  # [B, 32]
        # 潜在変数とラベル埋め込みを結合
        z_cat = torch.cat((z, label_embedding), dim=1)  # [B, latent_size+32]
        pred = self.decoder(z_cat)
        return pred, mu, logvar

# ------------------------------
# データロードと前処理
# ------------------------------
def load_data():
    """MNISTデータセットのロード"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root='./data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data/', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

# ------------------------------
# 損失関数
# ------------------------------
def loss_function(x, pred, mu, logvar):
    """再構成誤差とKLダイバージェンスの計算"""
    # 'sum' による総和になっているが必要に応じて 'mean' に変更することも検討
    recon_loss = F.mse_loss(pred, x, reduction='sum')
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
            y = y.to(device)
            # ワンホット化
            y_onehot = F.one_hot(y, num_classes=10).float()

            optimizer.zero_grad()
            pred, mu, logvar = model(x, y_onehot)
            recon_loss, kld = loss_function(x, pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optimizer.step()

            total_loss  += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kld   += kld.item() * x.size(0)
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
                y = y.to(device)
                y_onehot = F.one_hot(y, num_classes=10).float()

                pred, mu, logvar = model(x, y_onehot)
                recon_loss, kld = loss_function(x, pred, mu, logvar)
                loss = recon_loss + kld

                total_loss  += loss.item() * x.size(0)
                total_recon += recon_loss.item() * x.size(0)
                total_kld   += kld.item() * x.size(0)
                
                if i == 0:
                    plot(epoch, pred.cpu().numpy(), y.cpu().numpy())
            except Exception as e:
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

    dataset_size = len(test_loader.dataset)
    elapsed = time.time() - start_time
    return total_loss / dataset_size, total_kld / dataset_size, total_recon / dataset_size, elapsed

# ------------------------------
# 可視化と画像生成
# ------------------------------
def plot(epoch, pred, y, prefix='test_'):
    """生成された画像をグリッド表示して保存"""
    os.makedirs('./images', exist_ok=True)
    fig = plt.figure(figsize=(16, 16))
    for i in range(6):
        ax = fig.add_subplot(3, 2, i+1)
        ax.imshow(pred[i, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(str(y[i]))
    plt.savefig(f"./images/{prefix}epoch_{epoch+1}.jpg")
    plt.close()

def generate_image(epoch, z, y, model):
    """
    任意の潜在ベクトル z とラベル y（スカラー形式）を用いて画像生成する  
    y : テンソル [B]（0～9の整数）
    """
    with torch.no_grad():
        y_reshaped = y.view(-1, 1).float()  # 正規化せずそのままの値
        label_embedding = model.label_encoder(y_reshaped.to(device))
        z_cat = torch.cat((z.to(device), label_embedding), dim=1)
        pred = model.decoder(z_cat)
        plot(epoch, pred.cpu().numpy(), y.cpu().numpy(), prefix='Eval_')

def generate_all_digits_per_epoch(epoch, model, result_dir):
    """
    各数字(0-9)に対して1枚ずつ画像生成  
    各数字の生成には固定の潜在ベクトルとラベルからラベル埋め込みを計算する
    """
    model.eval()
    with torch.no_grad():
        np.random.seed(42)
        z = torch.randn(10, latent_size).to(device)
        digit_images = []
        for digit in range(10):
            digit_label = torch.tensor([[digit]], dtype=torch.float).to(device)
            label_embedding = model.label_encoder(digit_label)
            z_cat = torch.cat((z[digit].unsqueeze(0), label_embedding), dim=1)
            img = model.decoder(z_cat)
            digit_images.append(img[0, 0].cpu().numpy())
        
        fig = plt.figure(figsize=(20, 3))
        for i in range(10):
            ax = fig.add_subplot(1, 10, i+1)
            ax.imshow(digit_images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Digit {i}')
        plt.tight_layout()
        
        epoch_img_dir = os.path.join(result_dir, "epoch_images")
        os.makedirs(epoch_img_dir, exist_ok=True)
        file_name = os.path.join(epoch_img_dir, f"epoch_{epoch+1}.png")
        plt.savefig(file_name, dpi=200)
        plt.close()

def generate_all_digits(model, result_dir):
    """
    潜在空間からランダムにサンプルをとり，
    各数字（0～9）を3枚ずつ生成してグリッド表示する
    """
    print("数字画像を生成中...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(30, latent_size).to(device)
        labels = []
        for digit in range(10):
            labels.extend([digit] * 3)
        y = torch.tensor(labels, dtype=torch.float).to(device)
        digit_labels = y.view(-1, 1).float()  # 正規化せずそのままの値
        label_embedding = model.label_encoder(digit_labels)
        z_cat = torch.cat((z, label_embedding), dim=1)
        pred = model.decoder(z_cat)
        
        fig = plt.figure(figsize=(15, 5))
        for i in range(30):
            ax = fig.add_subplot(3, 10, i+1)
            ax.imshow(pred[i, 0].cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f"Digit: {int(y[i].item())}")
        plt.tight_layout()
        output_file = os.path.join(result_dir, 'generated_digits.png')
        plt.savefig(output_file, dpi=200)
        plt.close()
        print(f"生成された数字画像を保存しました: {output_file}")

# ------------------------------
# モデル保存と結果分析
# ------------------------------
def create_result_directory():
    """結果保存用のディレクトリ作成"""
    result_dir = "/home/1/uk05101/python_test/VAE_test/result_4"
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def save_model(model, epoch, result_dir=None):
    """エポックごとにモデルの状態を保存"""
    if result_dir is None:
        os.makedirs("./checkpoints", exist_ok=True)
        file_name = f'./checkpoints/model_{epoch+1}.pt'
    else:
        file_name = os.path.join(result_dir, f'model_{epoch+1}.pt')
    torch.save(model.state_dict(), file_name)
    if epoch == max_epoch - 1:
        final_name = os.path.join(result_dir, 'final_model.pt') if result_dir else './checkpoints/final_model.pt'
        torch.save(model.state_dict(), final_name)

def create_gif_from_epoch_images(result_dir, gif_name='training_progress.gif'):
    """各エポック画像からGIFを生成"""
    epoch_img_dir = os.path.join(result_dir, "epoch_images")
    files = sorted(
        [os.path.join(epoch_img_dir, f) for f in os.listdir(epoch_img_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    images = []
    for file in files:
        images.append(imageio.imread(file))
    gif_path = os.path.join(result_dir, gif_name)
    imageio.mimsave(gif_path, images, duration=0.5)
    print(f"GIFが保存されました: {gif_path}")

def plot_learning_curves(result_dir):
    """学習・評価の損失推移を可視化して保存"""
    try:
        train_loss = np.load(os.path.join(result_dir, "train_loss.npy"))
        test_loss  = np.load(os.path.join(result_dir, "test_loss.npy"))
        epochs = np.arange(1, train_loss.shape[0] + 1)
        
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
        plot_path = os.path.join(result_dir, 'learning_curves.png')
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Learning curves saved: {plot_path}")
    except Exception as e:
        print(f"Error occurred while plotting learning curves: {e}")
        traceback.print_exc()

# ------------------------------
# メイン処理
# ------------------------------
if __name__ == "__main__":
    result_dir = create_result_directory()
    
    train_loader, test_loader = load_data()
    print("データローダーを作成しました")
    
    model = CVAE(latent_size=latent_size).to(device)
    print("モデルを作成しました")
    
    if load_epoch > 0:
        checkpoint_path = f'./checkpoints/model_{load_epoch}.pt'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"モデル {load_epoch} をロードしました")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(load_epoch + 1, max_epoch):
        train_total, train_kld, train_recon, train_time = train(epoch, model, train_loader, optimizer)
        test_total, test_kld, test_recon, test_time = test(epoch, model, test_loader)
        
        if generate:
            z_sample = torch.randn(6, latent_size).to(device)
            y_sample = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
            generate_image(epoch, z_sample, y_sample, model)
        
        print(f"Epoch: {epoch+1}/{max_epoch} | 損失: {train_total:.4f} | 学習時間: {train_time:.2f}秒 / 評価時間: {test_time:.2f}秒")
        
        save_model(model, epoch, result_dir)
        train_loss_list.append([train_total, train_kld, train_recon])
        test_loss_list.append([test_total, test_kld, test_recon])
        np.save(os.path.join(result_dir, "train_loss.npy"), np.array(train_loss_list))
        np.save(os.path.join(result_dir, "test_loss.npy"), np.array(test_loss_list))
        
        generate_all_digits_per_epoch(epoch, model, result_dir)
    
    generate_all_digits(model, result_dir)
    plot_learning_curves(result_dir)
    create_gif_from_epoch_images(result_dir, gif_name='training_progress.gif')
    
    print(f"学習結果と生成画像は {result_dir} に保存されました")
