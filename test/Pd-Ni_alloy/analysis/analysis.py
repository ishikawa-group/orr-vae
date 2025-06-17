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
data_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/Pd-Ni_alloy/ccvae/data"

# プロット設定
plt.figure(figsize=(10, 6))

iterations = []
min_overpotentials = []

# iter0からiter4まで（またはiter5まで）のデータを取得
for iter_num in range(5):  # 0-4まで
    file_path = os.path.join(data_dir, f"iter{iter_num}_calculation_result.json")
    min_overpotential = load_min_overpotential(file_path)
    
    if min_overpotential is not None:
        iterations.append(iter_num)
        min_overpotentials.append(min_overpotential)

# プロット
if iterations:
    plt.plot(iterations, min_overpotentials, 'o-', 
            linewidth=2, markersize=8, 
            label='Minimum Overpotential')

# グラフの設定
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Minimum Overpotential (V)', fontsize=12)
plt.title('Minimum Overpotential vs Iteration', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(max(iterations) + 1))  # x軸の目盛りを整数に設定
plt.tight_layout()

# 出力ディレクトリの設定
output_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/Pd-Ni_alloy/analysis"
os.makedirs(output_dir, exist_ok=True)

# グラフを保存
output_path = os.path.join(output_dir, "min_overpotential_iteration.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# グラフを表示
plt.close()

# 数値データも表示
print("\nMinimum Overpotential Data:")
print("Iteration\tMin Overpotential (V)")
print("-" * 30)
for iter_num in range(5):
    file_path = os.path.join(data_dir, f"iter{iter_num}_calculation_result.json")
    min_overpotential = load_min_overpotential(file_path)
    if min_overpotential is not None:
        print(f"iter{iter_num}\t\t{min_overpotential:.3f}")
    else:
        print(f"iter{iter_num}\t\tN/A")