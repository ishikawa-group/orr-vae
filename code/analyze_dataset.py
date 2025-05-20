import os
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP
import statistics as stats
import numpy as np
from tqdm import tqdm  # 進捗バー
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ========== ユーザ定義 Dataset / util を import ==========
from catalyst_ccVAE import CatalystDataset, make_data_loaders
from tool import sort_atoms, slab_to_tensor
# ============================================================

# ---------- 入力パス ----------
struct_json = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/data/iter0_structure.json")
energy_json = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/data/iter0_reaction_energy.json")

# ---------- 出力パス ----------
result_dir = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/ccVAE")
out_path = result_dir / "input_data_analysis.txt"
result_dir.mkdir(parents=True, exist_ok=True)  # フォルダが無ければ作成

# ---------- データセット ----------
dataset = CatalystDataset(struct_json=struct_json,
                         energy_json=energy_json,
                         grid_size=(4, 4, 4))

loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# ---------- 集計用バッファ ----------
# 新しい値: 0=Pd, 1=Pt, 0.5=未使用位置
total_counts = {0: 0, 0.5: 0, 1: 0}
layer_counts = [{0: 0, 0.5: 0, 1: 0} for _ in range(4)]  # 4層

# 偶数層と奇数層を分けて集計するバッファ
even_layer_counts = {0: 0, 0.5: 0, 1: 0}  # 偶数層 (0, 2)
odd_layer_counts = {0: 0, 0.5: 0, 1: 0}   # 奇数層 (1, 3)

overpotentials = []

# ---------- 走査 ----------
with torch.no_grad():
    for slabs, ops in tqdm(loader, desc="Scanning dataset"):
        overpotentials.extend(ops.tolist())
        
        # (a) 全体の元素カウント
        values, counts = torch.unique(slabs, return_counts=True)
        for val, count in zip(values.tolist(), counts.tolist()):
            # 対応する辞書キーを更新
            key = round(val * 2) / 2  # 浮動小数点の丸め誤差対策
            total_counts[key] += count
            
        # (b) 層ごとのカウント
        for z in range(4):
            layer_data = slabs[:, z]
            values, counts = torch.unique(layer_data, return_counts=True)
            for val, count in zip(values.tolist(), counts.tolist()):
                key = round(val * 2) / 2
                layer_counts[z][key] += count
                
                # さらに偶数層・奇数層に分けて集計
                if z % 2 == 0:  # 偶数層 (0, 2)
                    even_layer_counts[key] += count
                else:  # 奇数層 (1, 3)
                    odd_layer_counts[key] += count

# ---------- 元素分布の可視化 ----------
def plot_element_distribution():
    labels = ['Pd (0)', 'Empty (0.5)', 'Pt (1)']
    
    # 全体の分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 全体の円グラフ
    values = [total_counts[0], total_counts[0.5], total_counts[1]]
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Element Distribution')
    
    # 層ごとの棒グラフ
    x = np.arange(4)
    width = 0.2
    
    pd_values = [layer_counts[z][0] for z in range(4)]
    empty_values = [layer_counts[z][0.5] for z in range(4)]
    pt_values = [layer_counts[z][1] for z in range(4)]
    
    ax2.bar(x - width, pd_values, width, label='Pd (0)')
    ax2.bar(x, empty_values, width, label='Empty (0.5)')
    ax2.bar(x + width, pt_values, width, label='Pt (1)')
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Count')
    ax2.set_title('Element Distribution by Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(result_dir / "element_distribution.png")
    plt.close()

# ---------- 過電圧分布の可視化 ----------
def plot_overpotential_distribution():
    plt.figure(figsize=(10, 6))
    plt.hist(overpotentials, bins=20, alpha=0.75, edgecolor='black')
    plt.xlabel('Overpotential (V)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Overpotentials')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_dir / "overpotential_distribution.png")
    plt.close()

# ---------- 統計値 ----------
op_arr = np.array(overpotentials)
summary = {
    "num_samples": len(dataset),
    "voxels_per_slab": int(np.prod(slabs.shape[1:]) if len(slabs) > 0 else 0),
    "op_mean": op_arr.mean(),
    "op_std": op_arr.std(ddof=0),
    "op_min": op_arr.min(),
    "op_max": op_arr.max(),
    "op_median": float(stats.median(overpotentials)),
}

# ---------- テキスト保存 ----------
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Input-Data Statistical Report\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total samples        : {summary['num_samples']}\n")
    f.write(f"Voxels / sample      : {summary['voxels_per_slab']}\n\n")

    f.write("=== Overpotential ===\n")
    for k in ("op_mean", "op_std", "op_min", "op_max", "op_median"):
        f.write(f"{k[3:].capitalize():>12}: {summary[k]:.4f}\n")
    f.write("\n")

    f.write("=== Element counts (global) ===\n")
    f.write("ID 0 (Pd)    : {}\n".format(total_counts[0]))
    f.write("ID 0.5 (Empty): {}\n".format(total_counts[0.5]))
    f.write("ID 1 (Pt)    : {}\n".format(total_counts[1]))
    f.write("\n")
    
    f.write("=== Element counts by layer type ===\n")
    f.write("Even layers (0, 2):\n")
    for k in [0, 0.5, 1]:
        f.write(f"  ID {k}: {even_layer_counts[k]}\n")
    f.write("\nOdd layers (1, 3):\n")
    for k in [0, 0.5, 1]: 
        f.write(f"  ID {k}: {odd_layer_counts[k]}\n")
    f.write("\n")

    f.write("=== Element counts per layer ===\n")
    for z, d in enumerate(layer_counts):
        f.write(f"Layer {z+1}:\n")
        for k in [0, 0.5, 1]:
            f.write(f"  ID {k}: {d[k]}\n")
        f.write("\n")

# ---------- 可視化の実行 ----------
try:
    plot_element_distribution()
    plot_overpotential_distribution()
    print(f"✓ 元素分布とヒストグラムを保存しました → {result_dir}")
except Exception as e:
    print(f"✗ グラフ作成中にエラーが発生しました: {e}")

print(f"✓ 統計情報を書き出しました → {out_path}")

def main():
    """メイン実行関数（コマンドラインからの実行用）"""
    print("データセット分析を完了しました")
    
if __name__ == "__main__":
    main()