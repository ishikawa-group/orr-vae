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

def load_structures_below_threshold(file_path, threshold=0.4):
    """JSONファイルから過電圧が閾値未満の構造を取得"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        structures = []
        for structure in data:
            if ('overpotential' in structure and 
                structure['overpotential'] is not None and 
                structure['overpotential'] < threshold and
                'ni_fraction' in structure):
                structures.append({
                    'ni_fraction': structure['ni_fraction'],
                    'overpotential': structure['overpotential'],
                    'unique_id': structure.get('unique_id', 'unknown')
                })
        
        return structures
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# ベースディレクトリ
data_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/Pd-Ni_alloy/ccvae/data"

# 出力ディレクトリの設定
output_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/Pd-Ni_alloy/analysis"
os.makedirs(output_dir, exist_ok=True)

# 新しいプロット: 過電圧<0.4の構造のNi含有量 vs 過電圧
plt.figure(figsize=(12, 8))

colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']
markers = ['o', 'o', 'o', 'o', 'o', 'o']

all_ni_fractions = []
all_overpotentials = []
all_iterations = []

for iter_num in range(6):
    file_path = os.path.join(data_dir, f"iter{iter_num}_calculation_result.json")
    structures = load_structures_below_threshold(file_path, threshold=0.4)
    
    if structures:
        ni_fractions = [s['ni_fraction'] for s in structures]
        overpotentials = [s['overpotential'] for s in structures]
        
        plt.scatter(ni_fractions, overpotentials, 
                   c=colors[iter_num], marker=markers[iter_num], 
                   s=60, alpha=0.7, label=f'iter{iter_num}')
        
        # 全体のデータにも追加
        all_ni_fractions.extend(ni_fractions)
        all_overpotentials.extend(overpotentials)
        all_iterations.extend([iter_num] * len(ni_fractions))
        
        print(f"\niter{iter_num}: {len(structures)} structures with overpotential < 0.4V")

# グラフの設定
plt.xlabel('Ni Fraction', fontsize=12)
plt.ylabel('Overpotential (V)', fontsize=12)
plt.title('Overpotential vs Ni Fraction for Structures with η < 0.4V', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='η = 0.4V threshold')
plt.tight_layout()

# グラフを保存
output_path_2 = os.path.join(output_dir, "ni_fraction_vs_overpotential.png")
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f"Ni fraction vs overpotential plot saved to: {output_path_2}")

plt.close()

# 統計情報の表示
if all_ni_fractions:
    print(f"\nTotal structures with overpotential < 0.4V: {len(all_ni_fractions)}")
    print(f"Ni fraction range: {min(all_ni_fractions):.3f} - {max(all_ni_fractions):.3f}")
    print(f"Overpotential range: {min(all_overpotentials):.3f} - {max(all_overpotentials):.3f}")
    
    # 最低過電圧の構造を特定
    min_idx = all_overpotentials.index(min(all_overpotentials))
    print(f"Best structure: Ni fraction = {all_ni_fractions[min_idx]:.3f}, "
          f"overpotential = {all_overpotentials[min_idx]:.3f}V "
          f"(from iter{all_iterations[min_idx]})")