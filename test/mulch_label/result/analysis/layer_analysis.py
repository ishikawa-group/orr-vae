import json
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

# --- 1. ファイルパスの設定 ---
overpotential_path = '/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/mulch_label/result/label-2/data/iter0_calculation_result.json'
structures_path = '/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/mulch_label/result/label-2/data/iter0_structures.json'
output_dir = '/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/test/mulch_label/result/label-2/figure'

# --- 2. データの読み込みと結合 ---
print("データの読み込みを開始します...")

# 過電圧データの読み込み
try:
    with open(overpotential_path, 'r') as f:
        overpotential_data = json.load(f)
    # unique_idをキー、過電圧を値とする辞書を作成
    overpotentials = {item['unique_id']: item['overpotential'] for item in overpotential_data}
    print(f"過電圧データを {len(overpotentials)} 件読み込みました。")
except FileNotFoundError:
    print(f"エラー: 過電圧ファイルが見つかりません: {overpotential_path}")
    exit()
except Exception as e:
    print(f"エラー: 過電圧ファイルの読み込み中にエラーが発生しました: {e}")
    exit()

# 構造データの読み込み (ASE DB形式のJSON)
try:
    structures = {}
    db = connect(structures_path)
    for row in db.select():
        structures[row.unique_id] = row.toatoms()
    print(f"構造データを {len(structures)} 件読み込みました。")
except FileNotFoundError:
    print(f"エラー: 構造ファイルが見つかりません: {structures_path}")
    exit()
except Exception as e:
    print(f"エラー: 構造ファイルの読み込み中にエラーが発生しました: {e}")
    exit()

# 過電圧と構造データをunique_idで結合
combined_data = []
for uid, atoms in structures.items():
    if uid in overpotentials:
        combined_data.append({
            'atoms': atoms,
            'overpotential': overpotentials[uid]
        })
print(f"過電圧と構造が紐付いたデータを {len(combined_data)} 件作成しました。")

# --- 3. 層ごとのPt原子数をカウント ---
print("層ごとのPt原子数をカウントしています...")

# 描画用データを格納する辞書
# キー: 層の名前, 値: (Pt原子数, 過電圧) のタプルのリスト
data_for_plotting = {
    'layer1': [],  # 最表面
    'layer2': [],
    'layer3': [],
    'layer4': []   # 最下層
}

for item in combined_data:
    atoms = item['atoms']
    overpotential = item['overpotential']

    # z座標に基づいて層を特定
    positions = atoms.get_positions()
    z_coords = positions[:, 2]
    
    # z座標のユニークな値を取得し、降順にソート (z座標大 = 表面)
    unique_z = sorted(np.unique(z_coords), reverse=True)

    # 4層構造であることを確認
    if len(unique_z) != 4:
        # print(f"警告: 構造 {item.get('unique_id', 'N/A')} は4層ではありません ({len(unique_z)}層)。スキップします。")
        continue

    # 各層のPt原子数をカウント
    pt_counts_per_layer = []
    for z_level in unique_z:
        # 現在のzレベルに属する原子のインデックスを取得
        layer_indices = np.where(np.abs(z_coords - z_level) < 1e-5)[0]
        count = 0
        for i in layer_indices:
            if atoms[i].symbol == 'Pt':
                count += 1
        pt_counts_per_layer.append(count)

    # データを層ごとに格納 (unique_zが降順なので、pt_counts_per_layer[0]がlayer1に対応)
    data_for_plotting['layer1'].append((pt_counts_per_layer[0], overpotential))
    data_for_plotting['layer2'].append((pt_counts_per_layer[1], overpotential))
    data_for_plotting['layer3'].append((pt_counts_per_layer[2], overpotential))
    data_for_plotting['layer4'].append((pt_counts_per_layer[3], overpotential))

print("カウントが完了しました。")

# --- 4. グラフの描画と保存 ---
print("グラフを作成しています...")

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# グラフの設定
plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3'] # 色覚多様性対応カラーセット
labels = ['Layer 1 (top surface)', 'Layer 2', 'Layer 3', 'Layer 4 (bottom)']
layer_keys = ['layer1', 'layer2', 'layer3', 'layer4']

# 各層のデータを散布図としてプロット
for i, key in enumerate(layer_keys):
    if data_for_plotting[key]:
        # (Pt数, 過電圧) のリストを、Pt数のリストと過電圧のリストに分解
        pt_counts, overpotentials_vals = zip(*data_for_plotting[key])
        ax.scatter(pt_counts, overpotentials_vals, label=labels[i], alpha=0.7, color=colors[i], s=50)

# 軸ラベルとタイトルの設定
ax.set_xlabel('Number of Pt atoms', fontsize=14)
ax.set_ylabel('Overpotential (eV)', fontsize=14)
ax.set_title('Relationship between Pt Atom Count and Overpotential in Each Layer', fontsize=16)

# 軸の範囲と目盛りの設定
ax.set_xticks(np.arange(0, 17, 1))
ax.tick_params(axis='both', which='major', labelsize=12)

# 凡例とグリッドの表示
ax.legend(title='Layer', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# グラフをファイルに保存
output_path = os.path.join(output_dir, 'pt_count_vs_overpotential_per_layer_iter0.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"グラフを {output_path} に保存しました。")