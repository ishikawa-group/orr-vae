import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# 自作モジュールのインポート
from catalyst_ccVAE import CVAE

# GPUが利用可能ならそれを使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

result_dir = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE")
model_path = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE/final_model.pt")

def load_model(model_path, latent_size=64):
    """学習済みモデルをロードする"""
    model = CVAE(latent_size=latent_size, condition_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def analyze_tensor(tensor):
    """テンソルの基本統計情報を取得（偶数層と奇数層を区別）"""
    # 新しいID分類
    # ID 0: 値 < 0.5
    # ID 1: 値 ≥ 0.5
    
    # 全体の統計
    id0_mask = tensor < 0.5
    id1_mask = tensor >= 0.5
    
    value_counts = {
        0: int(id0_mask.sum().item()),
        1: int(id1_mask.sum().item())
    }
    
    # 偶数層と奇数層の統計を別々に計算
    even_layers_stats = []  # 偶数層 (0, 2, 4, ...)
    odd_layers_stats = []   # 奇数層 (1, 3, 5, ...)
    
    # 各層ごとの統計情報
    layer_stats = []
    for i in range(tensor.size(0)):  # 各層について
        layer = tensor[i]
        
        # 偶数層と奇数層で異なる要素のみを抽出
        if i % 2 == 0:  # 偶数層
            # 偶数インデックスの要素のみを考慮
            elements = layer[0::2, 0::2]
        else:  # 奇数層
            # 奇数インデックスの要素のみを考慮
            elements = layer[1::2, 1::2]
        
        # 抽出した要素に対してID分類
        layer_id0_mask = elements < 0.5
        layer_id1_mask = elements >= 0.5
        
        layer_stats.append({
            'mean': elements.mean().item(),
            'std': elements.std().item(),
            'min': elements.min().item(),
            'max': elements.max().item(),
            'median': torch.median(elements).item(),
            # 新しいID分類でのカウント
            'id_counts': {
                0: int(layer_id0_mask.sum().item()),
                1: int(layer_id1_mask.sum().item())
            }
        })
        
        # 偶数層と奇数層の統計を別々に保存
        if i % 2 == 0:
            even_layers_stats.append(layer_stats[-1])
        else:
            odd_layers_stats.append(layer_stats[-1])
    
    return {
        'global_mean': tensor.mean().item(),
        'global_std': tensor.std().item(),
        'value_counts': value_counts,
        'layer_stats': layer_stats,
        'even_layers_stats': even_layers_stats,
        'odd_layers_stats': odd_layers_stats
    }

@torch.no_grad()
def test_conditional_outputs(model, result_dir):
    """異なる条件での出力テンソルをテストして結果を保存"""
    # 固定の潜在変数を使用（再現性のため）
    torch.manual_seed(42)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = result_dir / f"ccVAE_condition_test.txt"
    
    overpotentials = np.arange(0.2, 0.7, 0.1)  # 0.2から0.7まで0.1刻み
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"条件付きVAEモデル出力テスト - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"モデル: {model_path}\n")
        f.write(f"潜在変数サイズ: {model.latent_size}\n\n")
        f.write("テスト条件:\n")
        f.write(f"- 固定潜在変数: シード42\n")
        f.write(f"- 過電圧範囲: 0.2V から 0.7V\n\n")
        f.write("要素カウント基準:\n")
        f.write(f"- ID 0: 値 < 0.5\n")
        f.write(f"- ID 1: 値 ≥ 0.5\n\n")
        f.write("偶数層: 偶数インデックス要素 (0,2,4,...)\n")
        f.write("奇数層: 奇数インデックス要素 (1,3,5,...)\n\n")
        f.write("=" * 60 + "\n\n")
        
        all_tensors = []  # すべての出力テンソルを保存
        
        for op in overpotentials:
            print(f"過電圧 {op:.2f}V での出力をテスト中...")
            
            # 過電圧条件のエンコード
            y = torch.tensor([[op]], dtype=torch.float).to(device)
            label_embedding = model.label_encoder(y)
            
            # 潜在変数と条件の結合
            z = torch.randn(1, model.latent_size).to(device)
            z_cat = torch.cat((z, label_embedding), dim=1)
            
            # デコーダーを通して構造テンソルを生成
            tensor = model.decoder(z_cat)[0].cpu()
            
            all_tensors.append(tensor)
            
            # テンソルの分析
            stats = analyze_tensor(tensor)
            
            # 結果をテキストファイルに書き出し
            f.write(f"過電圧: {op:.2f}V\n")
            f.write(f"全体平均値: {stats['global_mean']:.4f}\n")
            f.write(f"全体標準偏差: {stats['global_std']:.4f}\n")
            f.write("元素分布 (ID: 個数):\n")
            for id_val, count in stats['value_counts'].items():
                f.write(f"  ID {id_val}: {count}個\n")
            
            # 偶数層と奇数層を別々に表示
            f.write("\n偶数層統計 (層 1, 3, 5, ...):\n")
            for i, layer_stat in enumerate(stats['even_layers_stats']):
                layer_idx = i * 2
                f.write(f"  層 {layer_idx+1}:\n")
                f.write(f"    平均値: {layer_stat['mean']:.4f}\n")
                f.write(f"    標準偏差: {layer_stat['std']:.4f}\n")
                f.write(f"    最小値: {layer_stat['min']:.4f}\n")
                f.write(f"    最大値: {layer_stat['max']:.4f}\n")
                f.write(f"    中央値: {layer_stat['median']:.4f}\n")
                f.write("    元素分布 (ID: 個数):\n")
                for id_val, count in layer_stat['id_counts'].items():
                    f.write(f"      ID {id_val}: {count}個\n")
            
            f.write("\n奇数層統計 (層 2, 4, 6, ...):\n")
            for i, layer_stat in enumerate(stats['odd_layers_stats']):
                layer_idx = i * 2 + 1
                f.write(f"  層 {layer_idx+1}:\n")
                f.write(f"    平均値: {layer_stat['mean']:.4f}\n")
                f.write(f"    標準偏差: {layer_stat['std']:.4f}\n")
                f.write(f"    最小値: {layer_stat['min']:.4f}\n")
                f.write(f"    最大値: {layer_stat['max']:.4f}\n")
                f.write(f"    中央値: {layer_stat['median']:.4f}\n")
                f.write("    元素分布 (ID: 個数):\n")
                for id_val, count in layer_stat['id_counts'].items():
                    f.write(f"      ID {id_val}: {count}個\n")
            
            # テンソル自体の値を出力
            f.write("\nテンソル値（生の出力）:\n")
            for layer_idx in range(tensor.size(0)):
                f.write(f"  層 {layer_idx+1} ({'偶数層' if layer_idx % 2 == 0 else '奇数層'}):\n")
                layer_data = tensor[layer_idx].numpy()
                # NumPyの配列表示オプションを設定
                np.set_printoptions(precision=3, suppress=True, threshold=1000)
                # 各行を文字列に変換して書き出し
                for row_idx in range(layer_data.shape[0]):
                    row_str = "    " + np.array2string(layer_data[row_idx], precision=3, suppress_small=True)
                    f.write(f"{row_str}\n")
                f.write("\n")
            
            f.write("\n" + "-" * 40 + "\n\n")

def plot_element_distribution(overpotentials, id_counts, result_dir):
    """過電圧による元素分布の変化をプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array(overpotentials)
    y0 = np.array([counts.get(0, 0) for counts in id_counts])  # ID 0 (< 0.5)
    y1 = np.array([counts.get(1, 0) for counts in id_counts])  # ID 1 (≥ 0.5)
    
    ax.plot(x, y0, 'b-', marker='o', label='ID 0 (< 0.5)')
    ax.plot(x, y1, 'r-', marker='s', label='ID 1 (≥ 0.5)')
    
    ax.set_xlabel('Overpotential (V)')
    ax.set_ylabel('Element Count')
    ax.set_title('Change in Element Distribution by Overpotential')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(result_dir / f"element_distribution.png")
    plt.close()

def main():
    """メイン実行関数"""
    print("条件付きVAEモデルのテストを開始します")
    
    # 結果保存用ディレクトリの確認・作成
    result_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # モデルのロード
        print(f"モデルをロードしています: {model_path}")
        model = load_model(model_path)
        print("モデルのロード完了")
        
        # 異なる条件での出力テスト
        print("異なる過電圧条件での出力テストを開始")
        test_conditional_outputs(model, result_dir)
        print("出力テスト完了")
        
        # データ集計とグラフ作成のために追加する場合
        # 例：特定の過電圧範囲でのテスト
        # overpotentials = np.arange(0.2, 0.7, 0.1)
        # id_counts = []
        # 
        # for op in overpotentials:
        #     # ここで各過電圧でのカウントを取得する処理
        #     # ...
        #     # id_counts.append(counts)
        # 
        # plot_element_distribution(overpotentials, id_counts, result_dir)
        
        print(f"結果は {result_dir} に保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()