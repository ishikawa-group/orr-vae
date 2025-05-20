import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 自作モジュールのインポート
from catalyst_ccVAE_2 import CVAE, make_data_loaders

# 必要に応じてUMAPをインポート（インストールされている場合）
try:
    import umap.umap_ as umap
    umap_available = True
except ImportError:
    umap_available = False
    print("UMAP is not available. Install with 'pip install umap-learn' if needed.")

# 潜在空間のデータを抽出する関数
def extract_latent_representations(model, train_loader, test_loader):
    """モデルの潜在空間表現を抽出する"""
    device = next(model.parameters()).device
    model.eval()
    
    # 潜在空間のエンコーディングと過電圧を保存するリスト
    all_mu = []
    all_overpotentials = []
    is_train = []  # 訓練データか否かのフラグ
    
    # 訓練データを処理
    print("Encoding training data...")
    with torch.no_grad():
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device).view(-1, 1).float()
            
            # エンコーダーで潜在表現を取得
            mu, _ = model.encoder(x, y)
            
            all_mu.append(mu.cpu().numpy())
            all_overpotentials.append(y.cpu().numpy())
            is_train.append(np.ones(len(x), dtype=bool))
    
    # テストデータを処理
    print("Encoding test data...")
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device).view(-1, 1).float()
            
            # エンコーダーで潜在表現を取得
            mu, _ = model.encoder(x, y)
            
            all_mu.append(mu.cpu().numpy())
            all_overpotentials.append(y.cpu().numpy())
            is_train.append(np.zeros(len(x), dtype=bool))
    
    # データの結合
    latent_vectors = np.vstack(all_mu)
    overpotentials = np.vstack(all_overpotentials).flatten()
    is_train = np.hstack(is_train)
    
    print(f"Extraction complete: {len(latent_vectors)} samples, latent dim: {latent_vectors.shape[1]}")
    
    return latent_vectors, overpotentials, is_train

# 次元削減して可視化する関数
def visualize_with_dim_reduction(latent_vectors, overpotentials, is_train, result_dir, 
                              method='tsne', perplexity=30, n_neighbors=15, min_dist=0.1, random_state=42):
    """次元削減アルゴリズムを用いて潜在ベクトルを2次元に削減し、可視化する"""
    method = method.lower()
    
    # 次元削減の実行
    if method == 'tsne':
        print(f"Running t-SNE dimension reduction (perplexity={perplexity})...")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        embedding_2d = reducer.fit_transform(latent_vectors)
        method_name = f"t-SNE (perplexity={perplexity})"
    elif method == 'umap' and umap_available:
        print(f"Running UMAP dimension reduction (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                         random_state=random_state)
        embedding_2d = reducer.fit_transform(latent_vectors)
        method_name = f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"
    else:
        print("Using t-SNE as fallback method.")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        embedding_2d = reducer.fit_transform(latent_vectors)
        method_name = f"t-SNE (perplexity={perplexity})"
    
    # DataFrameに変換
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'overpotential': overpotentials,
        'dataset': ['Train' if t else 'Test' for t in is_train]
    })
    
    # 1. 過電圧による色分けプロット
    plt.figure(figsize=(12, 10))
    
    plt.subplot(1, 1, 1)
    scatter = plt.scatter(df['x'], df['y'], c=df['overpotential'], 
                       cmap='viridis', alpha=0.8, s=60)
    cbar = plt.colorbar(scatter, label='Overpotential (V)')
    
    # 訓練・テストデータをマーカーで区別
    train_data = df[df['dataset'] == 'Train']
    test_data = df[df['dataset'] == 'Test']
    
    plt.scatter(train_data['x'], train_data['y'], facecolors='none', edgecolors='black', 
              alpha=0.4, s=100, label='Train')
    plt.scatter(test_data['x'], test_data['y'], facecolors='none', edgecolors='red', 
              alpha=0.4, s=100, label='Test')
    
    plt.title(f'Latent Space Visualization ({method_name})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(result_dir) / f'latent_space_{method}.png', dpi=300)
    print(f"Visualization saved: {Path(result_dir) / f'latent_space_{method}.png'}")
    
    # 2. 訓練・テストデータを別々に表示するプロット
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(train_data['x'], train_data['y'], c=train_data['overpotential'], 
                        cmap='viridis', alpha=0.8, s=60)
    plt.colorbar(scatter1, label='Overpotential (V)')
    plt.title(f'Train Data ({len(train_data)} samples)', fontsize=14)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(test_data['x'], test_data['y'], c=test_data['overpotential'], 
                        cmap='viridis', alpha=0.8, s=60)
    plt.colorbar(scatter2, label='Overpotential (V)')
    plt.title(f'Test Data ({len(test_data)} samples)', fontsize=14)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(result_dir) / f'latent_space_{method}_split.png', dpi=300)
    
    # データをCSVとして保存
    df.to_csv(Path(result_dir) / f'latent_space_{method}_data.csv', index=False)
    print(f"Data saved: {Path(result_dir) / f'latent_space_{method}_data.csv'}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='VAE latent space visualization')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'], 
                      help='Dimension reduction method (tsne or umap)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # GPUが利用可能ならそれを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パスの設定
    result_dir = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE")
    model_path = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE/final_model.pt")
    
    # データローダーを作成
    train_loader, test_loader = make_data_loaders(
        struct_json=Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/data/iter0_structure.json"),
        energy_json=Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/data/iter0_reaction_energy.json"),
        grid_size=(4, 4, 4),
        train_ratio=0.9,
        batch_size=args.batch_size,
        num_workers=4
    )
    print("Data loaders created")
    
    # モデルのロード
    latent_size = 64  # モデルの潜在変数サイズ
    model = CVAE(latent_size=latent_size, condition_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")
    
    # 潜在空間表現の抽出
    latent_vectors, overpotentials, is_train = extract_latent_representations(model, train_loader, test_loader)
    
    # 次元削減と可視化
    visualize_with_dim_reduction(
        latent_vectors, overpotentials, is_train, result_dir,
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed
    )
    
    print("Visualization complete.")

if __name__ == "__main__":
    main()