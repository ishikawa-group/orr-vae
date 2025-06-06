#!/usr/bin/env python3
"""
条件付きGANによるORR触媒構造生成
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ase import Atoms
from ase.data import atomic_numbers
from tool import slab_to_tensor, tensor_to_slab, convert_numpy_types

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CatalystDataset(Dataset):
    """触媒構造データセット"""
    
    def __init__(self, structures_file, overpotentials_file, grid_size=[4, 8, 8]):
        self.grid_size = grid_size
        self.structures = []
        self.overpotentials = []
        self.pt_fractions = []
        
        # 構造データの読み込み
        with open(structures_file, 'r') as f:
            structures_data = json.load(f)
        
        # 過電圧データの読み込み
        with open(overpotentials_file, 'r') as f:
            overpotentials_data = json.load(f)
        
        # unique_idをキーとした過電圧辞書を作成
        overpotential_dict = {
            item["unique_id"]: item["overpotential"] 
            for item in overpotentials_data
        }
        
        print(f"Loading {len(structures_data)} structures...")
        
        for key, structure_data in structures_data.items():
            unique_id = structure_data["unique_id"]
            
            if unique_id not in overpotential_dict:
                continue
            
            # ASE Atomsオブジェクトの再構築
            numbers = np.array(structure_data["numbers"])
            positions = np.array(structure_data["positions"])
            cell = np.array(structure_data["cell"])
            pbc = structure_data["pbc"]
            
            atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
            
            # テンソル変換
            try:
                tensor = slab_to_tensor(atoms, [4, 4, 4])  # 4x4x4 grid
                # 元素を0(空), 1(Ni), 2(Pt)にマッピング
                tensor_mapped = torch.zeros_like(tensor, dtype=torch.float32)
                tensor_mapped[tensor == atomic_numbers["Ni"]] = 1.0
                tensor_mapped[tensor == atomic_numbers["Pt"]] = 2.0
                
                # 4チャンネルに変換 (各層を1チャンネルとして扱う)
                tensor_4ch = tensor_mapped.view(4, 8, 8).float()
                
                self.structures.append(tensor_4ch)
                self.overpotentials.append(overpotential_dict[unique_id])
                self.pt_fractions.append(structure_data["pt_fraction"])
                
            except Exception as e:
                print(f"Error processing structure {unique_id}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.structures)} structures")
    
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        overpotential = self.overpotentials[idx]
        pt_fraction = self.pt_fractions[idx]
        
        # 条件ラベルの作成
        eta_label = 1.0 if overpotential < 0.5 else 0.0  # 過電圧0.5V未満なら1
        pt_label = 1.0 if pt_fraction < 0.5 else 0.0     # Pt含有量50%未満なら1
        
        return structure, torch.tensor([eta_label, pt_label], dtype=torch.float32)

class Generator(nn.Module):
    """生成器"""
    
    def __init__(self, noise_dim=128, condition_dim=2, channels=4):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        # 条件ラベルの埋め込み
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        
        # ノイズ+条件の線形変換
        self.linear = nn.Sequential(
            nn.Linear(noise_dim + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64 * 2 * 2),
            nn.BatchNorm1d(64 * 2 * 2),
            nn.ReLU(inplace=True)
        )
        
        # 転置畳み込み層
        self.deconv_layers = nn.Sequential(
            # 64x2x2 -> 32x4x4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x4x4 -> 16x8x8
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 16x8x8 -> 4x8x8
            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)  # 各ピクセルで確率分布
        )
    
    def forward(self, noise, conditions):
        # 条件の埋め込み
        condition_embed = self.condition_embed(conditions)
        
        # ノイズと条件を結合
        x = torch.cat([noise, condition_embed], dim=1)
        
        # 線形変換
        x = self.linear(x)
        x = x.view(x.size(0), 64, 2, 2)
        
        # 転置畳み込み
        x = self.deconv_layers(x)
        
        return x

class Discriminator(nn.Module):
    """識別器"""
    
    def __init__(self, channels=4):
        super(Discriminator, self).__init__()
        
        # 畳み込み層
        self.conv_layers = nn.Sequential(
            # 4x8x8 -> 16x4x4
            nn.Conv2d(channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x4x4 -> 32x2x2
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x2x2 -> 64x1x1
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 出力層: 3ノード（真偽, ORR過電圧ラベル, Pt含有量ラベル）
        self.output_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layers(x)
        return x

def train_gan(train_loader, num_epochs=100, lr=0.0002, beta1=0.5, save_dir="./models"):
    """GANの学習"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # モデルの初期化
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # オプティマイザ
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # 損失関数
    criterion = nn.BCELoss()
    
    # 学習ループ
    losses_G = []
    losses_D = []
    
    for epoch in range(num_epochs):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for batch_idx, (real_data, real_conditions) in enumerate(train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            real_conditions = real_conditions.to(device)
            
            # 真偽ラベル
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # === 識別器の学習 ===
            optimizer_D.zero_grad()
            
            # 真データの識別
            real_output = discriminator(real_data)
            real_validity = real_output[:, 0:1]  # 真偽判定
            real_eta_pred = real_output[:, 1:2]  # ORR過電圧予測
            real_pt_pred = real_output[:, 2:3]   # Pt含有量予測
            
            # 真データの損失
            d_loss_real_validity = criterion(real_validity, real_labels)
            d_loss_real_eta = criterion(real_eta_pred, real_conditions[:, 0:1])
            d_loss_real_pt = criterion(real_pt_pred, real_conditions[:, 1:2])
            d_loss_real = d_loss_real_validity + d_loss_real_eta + d_loss_real_pt
            
            # 偽データの生成と識別
            noise = torch.randn(batch_size, 128, device=device)
            target_conditions = torch.ones(batch_size, 2, device=device)  # 目標条件：両方1
            fake_data = generator(noise, target_conditions)
            fake_output = discriminator(fake_data.detach())
            fake_validity = fake_output[:, 0:1]
            
            d_loss_fake = criterion(fake_validity, fake_labels)
            
            # 識別器の総損失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # === 生成器の学習 ===
            optimizer_G.zero_grad()
            
            # 生成器による偽データの生成
            fake_output = discriminator(fake_data)
            fake_validity = fake_output[:, 0:1]
            fake_eta_pred = fake_output[:, 1:2]
            fake_pt_pred = fake_output[:, 2:3]
            
            # 生成器の損失（真データとして認識され、目標条件を満たすことを目指す）
            g_loss_validity = criterion(fake_validity, real_labels)
            g_loss_eta = criterion(fake_eta_pred, target_conditions[:, 0:1])
            g_loss_pt = criterion(fake_pt_pred, target_conditions[:, 1:2])
            g_loss = g_loss_validity + g_loss_eta + g_loss_pt
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_loss_G += g_loss.item()
            epoch_loss_D += d_loss.item()
        
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        
        losses_G.append(avg_loss_G)
        losses_D.append(avg_loss_D)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - G Loss: {avg_loss_G:.4f}, D Loss: {avg_loss_D:.4f}")
    
    # モデルの保存
    torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))
    
    # 学習曲線の保存
    plt.figure(figsize=(10, 5))
    plt.plot(losses_G, label='Generator Loss')
    plt.plot(losses_D, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()
    
    print(f"Models saved to {save_dir}")
    return generator, discriminator

def main():
    parser = argparse.ArgumentParser(description="Train conditional GAN for catalyst generation")
    parser.add_argument("--structures_file", type=str, required=True,
                        help="Input structures JSON file")
    parser.add_argument("--overpotentials_file", type=str, required=True,
                        help="Input overpotentials JSON file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./models",
                        help="Directory to save models")
    
    args = parser.parse_args()
    
    # データセットの作成
    dataset = CatalystDataset(args.structures_file, args.overpotentials_file)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # GANの学習
    train_gan(train_loader, num_epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)

if __name__ == "__main__":
    main()