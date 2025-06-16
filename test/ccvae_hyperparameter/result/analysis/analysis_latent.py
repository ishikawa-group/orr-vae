import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_min_overpotential(file_path):
    """JSONファイルから最小過電圧を取得"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 全ての過電圧値を取得
        overpotentials = []
        for structure in data:
            if 'overpotential' in structure and structure['overpotential'] is not None:
                overpotentials.append(structure['overpotential'])
        
        return min(overpotentials) if overpotentials else None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# ベースディレクトリ
base_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/ccvae_hyperparameter/result/latent_size"

# latent_size値のリスト
latent_values = ["8", "16", "32", "64", "128", "256"]
latent_labels = ["latent = 8", "latent = 16", "latent = 32", "latent = 64", "latent = 128", "latent = 256"]

# カラーマップ
colors = plt.cm.plasma(np.linspace(0, 1, len(latent_values)))

# プロット設定
plt.figure(figsize=(10, 6))

for i, (latent, label) in enumerate(zip(latent_values, latent_labels)):
    iterations = []
    min_overpotentials = []
    
    # iter0からiter4まで（またはiter5まで）のデータを取得
    for iter_num in range(6):  # 0-5まで
        file_path = os.path.join(base_dir, f"latent_size_{latent}", "data", f"iter{iter_num}_calculation_result.json")
        min_overpotential = load_min_overpotential(file_path)
        
        if min_overpotential is not None:
            iterations.append(iter_num)
            min_overpotentials.append(min_overpotential)
    
    # プロット
    if iterations:
        plt.plot(iterations, min_overpotentials, 'o-', 
                color=colors[i], linewidth=2, markersize=6, 
                label=label)

# グラフの設定
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Minimum Overpotential (V)', fontsize=12)
plt.title('Minimum Overpotential vs Iteration for Different Latent Dimensions', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 出力ディレクトリの設定
output_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/ccvae_hyperparameter/result/analysis"
os.makedirs(output_dir, exist_ok=True)

# グラフを保存
output_path = os.path.join(output_dir, "min_overpotential_comparison_latent_size.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# グラフを表示
plt.close()

# 数値データも表示
print("\nMinimum Overpotential Data:")
print("Latent\t", end="")
for i in range(6):
    print(f"iter{i}\t", end="")
print()

for i, (latent, label) in enumerate(zip(latent_values, latent_labels)):
    print(f"{label}\t", end="")
    for iter_num in range(6):
        file_path = os.path.join(base_dir, f"latent_size_{latent}", "data", f"iter{iter_num}_calculation_result.json")
        min_overpotential = load_min_overpotential(file_path)
        if min_overpotential is not None:
            print(f"{min_overpotential:.3f}\t", end="")
        else:
            print("N/A\t", end="")
    print()