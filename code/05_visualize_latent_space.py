import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from pathlib import Path
import argparse

from tool import make_data_loaders_from_json

def parse_args():
    parser = argparse.ArgumentParser(description="潜在空間可視化")
    parser.add_argument("--iter", type=int, default=0,
                       help="可視化したいiterのモデル (default: 0)")
    parser.add_argument("--latent_size", type=int, default=32,
                       help="潜在変数の次元 (default: 32)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="バッチサイズ (default: 32)")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).parent / "data"),
                       help="データディレクトリのパス (default: ./data)")
    parser.add_argument("--result_dir", type=str, default=str(Path(__file__).parent / "result"),
                       help="結果ディレクトリのパス (default: ./result)")
    return parser.parse_args()

def load_vae_class():
    """03_conditional_vae.pyからConditionalVAEクラスを動的にインポート"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vae_script_path = os.path.join(current_dir, "03_conditional_vae.py")
    
    if not os.path.exists(vae_script_path):
        raise FileNotFoundError(f"VAEスクリプトが見つかりません: {vae_script_path}")
    
    print(f"VAEスクリプトを読み込み中: {vae_script_path}")
    
    spec = importlib.util.spec_from_file_location("conditional_vae", vae_script_path)
    conditional_vae_module = importlib.util.module_from_spec(spec)
    sys.modules["conditional_vae"] = conditional_vae_module
    spec.loader.exec_module(conditional_vae_module)
    return conditional_vae_module.ConditionalVAE

def encode_all_data_with_raw_overpotentials(model, dataset, batch_size=32):
    """
    データセット全体を潜在変数にエンコードし、元の過電圧値も取得
    平均値（μ）のみを使用（決定論的な表現）
    
    Returns:
        latent_vectors: numpy array of shape (N, latent_size) - μの値
        raw_overpotentials: numpy array of shape (N,) - 元の過電圧値
        binary_labels: numpy array of shape (N, 2) - 2つの条件ラベル [overpotential_label, pt_fraction_label]
    """
    model.eval()
    latent_vectors = []
    raw_overpotentials = []
    binary_labels = []
    
    # データセット全体を順次処理
    with torch.no_grad():
        for i in range(len(dataset)):
            if i % 100 == 0:
                print(f"エンコード中... {i+1}/{len(dataset)}")
            
            # 個別のデータを取得
            data, binary_label = dataset[i]
            
            # 元の過電圧値を取得
            raw_overpotential = dataset.get_raw_overpotential(i)
            
            # バッチ次元を追加してデバイスに移動
            data = data.unsqueeze(0).to(device).float()
            # binary_labelは[overpotential_label, pt_fraction_label]の2次元配列
            binary_label_tensor = torch.tensor(binary_label).unsqueeze(0).to(device).float()  # [1, 2]
            
            # エンコードして平均と分散を取得
            mu, logvar = model.encode(data, binary_label_tensor)
            
            # 平均値（μ）のみを使用（決定論的な潜在表現）
            z = mu
            
            latent_vectors.append(z.cpu().numpy())
            raw_overpotentials.append(raw_overpotential)
            binary_labels.append(binary_label)
    
    # リストを結合
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    raw_overpotentials = np.array(raw_overpotentials)
    binary_labels = np.array(binary_labels)
    
    return latent_vectors, raw_overpotentials, binary_labels

def visualize_latent_space_tsne(latent_vectors, raw_overpotentials, binary_labels, save_path_base):
    """
    t-SNEで潜在空間を2次元に可視化
    2つの別々の画像ファイルを出力
    1. 全データの連続値ヒートマップ
    2. 過電圧0.5以下のデータのみを赤青で表示
    """
    print("t-SNE実行中...")
    print(f"潜在変数の形状: {latent_vectors.shape}")
    print(f"潜在変数の統計: 平均={latent_vectors.mean():.3f}, 標準偏差={latent_vectors.std():.3f}")
    
    # t-SNEで2次元に削減
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=min(30, len(latent_vectors) // 3),
        n_iter=1000,
        verbose=1
    )
    points_2d = tsne.fit_transform(latent_vectors)
    
    # 画像1: 全データの連続値の過電圧でカラーマップ
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(
        points_2d[:, 0], 
        points_2d[:, 1], 
        c=raw_overpotentials, 
        cmap='viridis_r',
        s=40,
        alpha=0.8
    )
    colorbar1 = plt.colorbar(scatter1)
    colorbar1.set_label('Overpotential (V)', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('All Data\n(Colored by Overpotential)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 保存パスを作成（拡張子を除去してサフィックス追加）
    save_path_all = save_path_base.replace('.png', '_all_data.png')
    plt.tight_layout()
    plt.savefig(save_path_all, dpi=300, bbox_inches='tight')
    print(f"全データの可視化結果を保存: {save_path_all}")
    plt.close()
    
    # 画像2: 過電圧0.5以下のデータのみ表示
    plt.figure(figsize=(10, 8))
    
    # 過電圧0.5以下のデータをフィルタリング
    mask_low_overpotential = raw_overpotentials <= 0.5
    
    if np.any(mask_low_overpotential):
        points_filtered = points_2d[mask_low_overpotential]
        overpotentials_filtered = raw_overpotentials[mask_low_overpotential]
        
        # 赤青のカラーマップで表示
        scatter2 = plt.scatter(
            points_filtered[:, 0], 
            points_filtered[:, 1], 
            c=overpotentials_filtered, 
            cmap='coolwarm_r',
            s=40,
            alpha=0.8
        )
        colorbar2 = plt.colorbar(scatter2)
        colorbar2.set_label('Overpotential (V)', fontsize=14)
        
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.title(f'High Performance Data (η ≤ 0.5V)\nn = {np.sum(mask_low_overpotential)}', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        print(f"過電圧0.5V以下のデータ数: {np.sum(mask_low_overpotential)}")
        print(f"フィルタ後の過電圧範囲: {overpotentials_filtered.min():.3f} ~ {overpotentials_filtered.max():.3f} V")
    else:
        plt.text(0.5, 0.5, 'No data with η ≤ 0.5V', 
                transform=plt.gca().transAxes, 
                ha='center', va='center', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.title('High Performance Data (η ≤ 0.5V)\nn = 0', fontsize=16)
        print("警告: 過電圧0.5V以下のデータが見つかりませんでした")
    
    save_path_filtered = save_path_base.replace('.png', '_high_performance.png')
    plt.tight_layout()
    plt.savefig(save_path_filtered, dpi=300, bbox_inches='tight')
    print(f"高性能データの可視化結果を保存: {save_path_filtered}")
    plt.close()
    
    # 統計情報を出力
    print(f"\n=== 統計情報 ===")
    print(f"全データ数: {len(raw_overpotentials)}")
    print(f"潜在変数の次元: {latent_vectors.shape[1]}")
    print(f"潜在変数統計: 平均={latent_vectors.mean():.3f}, 標準偏差={latent_vectors.std():.3f}")
    print(f"潜在変数範囲: {latent_vectors.min():.3f} ~ {latent_vectors.max():.3f}")
    print(f"全データのOverpotential範囲: {raw_overpotentials.min():.3f} ~ {raw_overpotentials.max():.3f} V")
    print(f"全データのOverpotential平均: {raw_overpotentials.mean():.3f} V")
    print(f"全データのOverpotential標準偏差: {raw_overpotentials.std():.3f} V")
    
    # 過電圧0.5V以下のデータの統計
    if np.any(mask_low_overpotential):
        print(f"高性能触媒（η ≤ 0.5V）の数: {np.sum(mask_low_overpotential)}")
        print(f"高性能触媒の割合: {np.sum(mask_low_overpotential)/len(raw_overpotentials)*100:.1f}%")
    
    print(f"高性能触媒（ラベル1）の数: {np.sum(binary_labels == 1)}")
    print(f"低性能触媒（ラベル0）の数: {np.sum(binary_labels == 0)}")

def main():
    args = parse_args()
    
    # パス設定（相対パス）
    BASE_DATA_PATH = args.data_dir
    RESULT_DIR = args.result_dir
    MODEL_PATH = f"{RESULT_DIR}/iter{args.iter}/final_cvae_iter{args.iter}.pt"
    
    STRUCTURES_DB_PATHS = [
        f"{BASE_DATA_PATH}/iter{i}_structures.json" for i in range(args.iter + 1)
    ]
    
    OVERPOTENTIALS_JSON_PATHS = [
        f"{BASE_DATA_PATH}/iter{i}_calculation_result.json" for i in range(args.iter + 1)
    ]
    
    # デバイス設定
    if torch.cuda.is_available():
        global device
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用デバイス: {device}")
    
    print(f"=== 潜在空間可視化 (iter{args.iter}) - 平均値版 ===")
    print(f"データディレクトリ: {BASE_DATA_PATH}")
    print(f"結果ディレクトリ: {RESULT_DIR}")
    print(f"モデルパス: {MODEL_PATH}")
    print(f"潜在変数次元: {args.latent_size}")
    
    # ConditionalVAEクラスを動的に読み込み
    ConditionalVAE = load_vae_class()
    
    # データセットの作成（全データを使用）
    train_loader, test_loader, dataset = make_data_loaders_from_json(
        structures_db_paths=STRUCTURES_DB_PATHS,
        overpotentials_json_paths=OVERPOTENTIALS_JSON_PATHS,
        use_binary_labels=True,  # これはラベルの形式であり、元データは保持される
        train_ratio=1.0,  # 全データを使用
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
        grid_size=[4, 4, 4]
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    
    # 過電圧の統計情報を表示
    stats = dataset.get_overpotential_stats()
    print(f"過電圧統計: 平均={stats['mean']:.3f}V, 範囲={stats['min']:.3f}~{stats['max']:.3f}V")
    
    # モデルの読み込み
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        return
    
    model = ConditionalVAE(latent_size=args.latent_size, condition_dim=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"モデルを読み込みました: {MODEL_PATH}")
    
    # 全データをエンコード（平均値版）
    print("注意: 平均値（μ）のみを使用した決定論的な潜在変数を使用します")
    latent_vectors, raw_overpotentials, binary_labels = encode_all_data_with_raw_overpotentials(
        model, dataset, args.batch_size
    )
    
    print(f"エンコード完了: {latent_vectors.shape[0]}個のデータ")
    print(f"潜在変数次元: {latent_vectors.shape[1]}")
    
    # 保存先ディレクトリ作成
    vis_dir = f"{RESULT_DIR}/visualization/iter{args.iter}"
    os.makedirs(vis_dir, exist_ok=True)
    
    # t-SNE可視化（2つのプロットを含む）
    visualize_latent_space_tsne(
        latent_vectors, 
        raw_overpotentials,
        binary_labels,
        f"{vis_dir}/tsne_latent_space_iter{args.iter}_mean.png"
    )
    
    # データも保存
    np.save(f"{vis_dir}/latent_vectors_iter{args.iter}_mean.npy", latent_vectors)
    np.save(f"{vis_dir}/raw_overpotentials_iter{args.iter}.npy", raw_overpotentials)
    np.save(f"{vis_dir}/binary_labels_iter{args.iter}.npy", binary_labels)
    
    print(f"可視化結果を保存しました: {vis_dir}")
    print("ファイル名に '_mean' が付いているものが平均値版です")

if __name__ == "__main__":
    main()