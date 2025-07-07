import json
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')  # バックエンドを非インタラクティブに設定
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import argparse

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ORR触媒データの分析とプロット作成')
    parser.add_argument('--iter', type=int, default=4,
                       help='最大イテレーション番号 (指定した番号以下のiterを分析, default: 4)')
    parser.add_argument('--base_path', type=str, 
                       default=str(Path(__file__).parent / "data"),
                       help='データディレクトリのパス')
    parser.add_argument('--output_path', type=str,
                       default=str(Path(__file__).parent / "result" / "figure"),
                       help='出力ディレクトリのパス')
    return parser.parse_args()

def load_json_data(file_path):
    """JSONファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_orr_catalyst_data(max_iter=4, base_path=None, output_path=None):
    """ORR触媒データの分析とプロット作成"""
    
    # データファイルのパス設定（相対パス使用）
    if base_path is None:
        base_path = Path(__file__).parent / "data"
    else:
        base_path = Path(base_path)
    
    if output_path is None:
        output_path = Path(__file__).parent / "result" / "figure"
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== ORR触媒データ分析 (iter0 ~ iter{max_iter}) ===")
    print(f"データディレクトリ: {base_path}")
    print(f"出力ディレクトリ: {output_path}")
    
    # iter0からmax_iterまでのデータを読み込み
    all_data = {}
    iter_range = list(range(max_iter + 1))  # 0からmax_iterまで
    
    for iter_num in iter_range:
        file_path = base_path / f"iter{iter_num}_calculation_result.json"
        if file_path.exists():
            try:
                data = load_json_data(file_path)
                all_data[f"iter{iter_num}"] = data
                print(f"iter{iter_num}: {len(data)} samples loaded")
            except Exception as e:
                print(f"Error loading iter{iter_num}: {e}")
        else:
            print(f"Warning: {file_path} not found")
    
    if not all_data:
        print("No data files found!")
        return None
    
    print(f"Successfully loaded data from {len(all_data)} iterations")
    
    # データをDataFrameに変換
    df_list = []
    for iter_name, data in all_data.items():
        df = pd.DataFrame(data)
        df['iter'] = iter_name
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} total samples")
    print(f"Iter distribution: {combined_df['iter'].value_counts().sort_index()}")
    
    # カラーパレットの設定（最大10色まで対応）
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {f'iter{i}': mcolors.to_hex(tab10_colors[i % 10]) for i in range(max_iter + 1)}
    
    # 全データのレンジを計算してヒストグラムの範囲を統一
    ni_fraction_clean = combined_df['ni_fraction'].dropna()
    overpotential_clean = combined_df['overpotential'].dropna()
    
    ni_range = (ni_fraction_clean.min(), ni_fraction_clean.max())
    overpot_range = (overpotential_clean.min(), overpotential_clean.max())
    
    print(f"Data ranges - Ni fraction: {ni_range[0]:.3f} ~ {ni_range[1]:.3f}")
    print(f"Data ranges - Overpotential: {overpot_range[0]:.3f} ~ {overpot_range[1]:.3f} V")
    
    # 1. Ni fraction のヒストグラム
    plt.figure(figsize=(12, 7))
    
    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        # NaNを除外してヒストグラムを作成
        ni_fraction_clean = iter_data['ni_fraction'].dropna()
        plt.hist(ni_fraction_clean, 
                bins=30, 
                range=ni_range,
                alpha=0.5, 
                label=f'{iter_name} (n={len(ni_fraction_clean)})', 
                color=colors.get(iter_name, 'gray'),
                edgecolor='black',
                linewidth=0.5)
    
    plt.xlabel('Ni Fraction', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Ni Fraction (iter0 ~ iter{max_iter})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # ヒストグラムを保存
    hist_output = output_path / f"ni_fraction_histogram_iter0-{max_iter}.png"
    plt.savefig(hist_output, dpi=300, bbox_inches='tight')
    print(f"Histogram saved: {hist_output}")
    plt.close()
    
    # 2. Overpotential のヒストグラム
    plt.figure(figsize=(12, 7))
    
    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        # NaNを除外してヒストグラムを作成
        overpotential_clean = iter_data['overpotential'].dropna()
        plt.hist(overpotential_clean, 
                bins=30, 
                range=overpot_range,
                alpha=0.5, 
                label=f'{iter_name} (n={len(overpotential_clean)})', 
                color=colors.get(iter_name, 'gray'),
                edgecolor='black',
                linewidth=0.5)
    
    plt.xlabel('Overpotential (V)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Overpotential (iter0 ~ iter{max_iter})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Overpotentialヒストグラムを保存
    overpot_hist_output = output_path / f"overpotential_histogram_iter0-{max_iter}.png"
    plt.savefig(overpot_hist_output, dpi=300, bbox_inches='tight')
    print(f"Overpotential histogram saved: {overpot_hist_output}")
    plt.close()
    
    # 3. Overpotential vs Ni fraction のscatter plot
    plt.figure(figsize=(12, 8))
    
    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        # NaNを除外
        valid_data = iter_data.dropna(subset=['ni_fraction', 'overpotential'])
        plt.scatter(valid_data['ni_fraction'], 
                   valid_data['overpotential'],
                   alpha=0.6,
                   label=f'{iter_name} (n={len(valid_data)})',
                   color=colors.get(iter_name, 'gray'),
                   s=40,
                   edgecolors='black',
                   linewidths=0.3)
    
    plt.xlabel('Ni Fraction', fontsize=12)
    plt.ylabel('Overpotential (V)', fontsize=12)
    plt.title(f'Overpotential vs Ni Fraction (iter0 ~ iter{max_iter})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Scatter plotを保存
    scatter_output = output_path / f"overpotential_vs_ni_fraction_iter0-{max_iter}.png"
    plt.savefig(scatter_output, dpi=500, bbox_inches='tight')
    print(f"Scatter plot saved: {scatter_output}")
    plt.close()
    
    # 4. 統計サマリー
    print(f"\n=== Statistical Summary (iter0 ~ iter{max_iter}) ===")
    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        ni_fraction_clean = iter_data['ni_fraction'].dropna()
        overpotential_clean = iter_data['overpotential'].dropna()
        
        print(f"\n{iter_name}:")
        print(f"  Total samples: {len(iter_data)}")
        print(f"  Ni fraction samples: {len(ni_fraction_clean)}")
        print(f"  Overpotential samples: {len(overpotential_clean)}")
        
        if len(ni_fraction_clean) > 0:
            print(f"  Ni fraction - Mean: {ni_fraction_clean.mean():.3f}, Std: {ni_fraction_clean.std():.3f}")
        
        if len(overpotential_clean) > 0:
            print(f"  Overpotential - Mean: {overpotential_clean.mean():.3f}, Std: {overpotential_clean.std():.3f}")
            print(f"  Min overpotential: {overpotential_clean.min():.3f}")
            
            # 最小overpotentialのNi fractionを取得
            min_idx = iter_data['overpotential'].idxmin()
            if pd.notna(iter_data.loc[min_idx, 'ni_fraction']):
                print(f"    (Ni fraction: {iter_data.loc[min_idx, 'ni_fraction']:.3f})")
    
    # 5. ボックスプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Ni fractionのボックスプロット
    valid_ni_data = combined_df.dropna(subset=['ni_fraction'])
    if not valid_ni_data.empty:
        # seabornを使用してカラフルなボックスプロットを作成
        iter_order = sorted(valid_ni_data['iter'].unique(), key=lambda x: int(x.replace('iter', '')))
        palette = [colors.get(iter_name, 'gray') for iter_name in iter_order]
        
        sns.boxplot(data=valid_ni_data, x='iter', y='ni_fraction', ax=ax1, 
                   order=iter_order, palette=palette, hue='iter', legend=False)
        ax1.set_title(f'Ni Fraction Distribution (iter0 ~ iter{max_iter})')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Ni Fraction')
        ax1.grid(True, alpha=0.3)
    
    # Overpotentialのボックスプロット
    valid_overpot_data = combined_df.dropna(subset=['overpotential'])
    if not valid_overpot_data.empty:
        # seabornを使用してカラフルなボックスプロットを作成
        iter_order = sorted(valid_overpot_data['iter'].unique(), key=lambda x: int(x.replace('iter', '')))
        palette = [colors.get(iter_name, 'gray') for iter_name in iter_order]
        
        sns.boxplot(data=valid_overpot_data, x='iter', y='overpotential', ax=ax2,
                   order=iter_order, palette=palette, hue='iter', legend=False)
        ax2.set_title(f'Overpotential Distribution (iter0 ~ iter{max_iter})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Overpotential (V)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ボックスプロットを保存
    boxplot_output = output_path / f"boxplots_comparison_iter0-{max_iter}.png"
    plt.savefig(boxplot_output, dpi=300, bbox_inches='tight')
    print(f"Boxplots saved: {boxplot_output}")
    plt.close()
    
    # 6. データの詳細分析（CSV出力）
    try:
        summary_stats = combined_df.groupby('iter').agg({
            'ni_fraction': ['count', 'mean', 'std', 'min', 'max'],
            'overpotential': ['count', 'mean', 'std', 'min', 'max'],
            'pt_fraction': ['count', 'mean', 'std'],
            'limiting_potential': ['count', 'mean', 'std']
        }).round(4)
        
        summary_output = output_path / f"summary_statistics_iter0-{max_iter}.csv"
        summary_stats.to_csv(summary_output)
        print(f"Summary statistics saved: {summary_output}")
    except Exception as e:
        print(f"Warning: Could not save summary statistics: {e}")
    
    # 7. 構造の可視化（iterごとの変化）
    structure_data = {}
    for iter_num in iter_range:
        structure_file_path = base_path / f"iter{iter_num}_structures.json"
        if structure_file_path.exists():
            try:
                with open(structure_file_path, 'r', encoding='utf-8') as f:
                    structures = json.load(f)
                structure_data[f"iter{iter_num}"] = structures
                print(f"iter{iter_num}: {len(structures)} structures loaded")
            except Exception as e:
                print(f"Error loading structures for iter{iter_num}: {e}")
    
    if structure_data:
        # 構造可視化の作成
        create_structure_visualization(structure_data, output_path, max_iter)
    
    return combined_df

def create_structure_visualization(structure_data, output_path, max_iter):
    """構造の可視化を作成"""
    print(f"\n=== Creating Structure Visualization ===")
    
    # 全ての構造を取得（各iterから全てのサンプル）
    all_structures = []
    for iter_name, structures in structure_data.items():
        iter_num = int(iter_name.replace('iter', ''))
        for struct_key, structure in structures.items():
            # 数値キーのみを処理（"ids"などの非数値キーを除外）
            try:
                struct_num = int(struct_key)
                all_structures.append({
                    'iter': iter_num,
                    'iter_name': iter_name,
                    'struct_key': struct_key,
                    'struct_num': struct_num,
                    'structure': structure
                })
            except ValueError:
                # 数値でないキーはスキップ
                continue
    
    if not all_structures:
        print("No structures found for visualization")
        return
    
    # iterとstruct_keyでソート
    all_structures.sort(key=lambda x: (x['iter'], x['struct_num']))
    
    # 構造の可視化用のデータを準備
    atom_colors = {28: '#90EE90', 78: '#C0C0C0'}  # Ni: 薄緑, Pt: シルバー
    
    # 全構造を処理してモザイク用の配列を作成
    mosaic_data = []
    y_offset = 0
    layer_positions = [0, 16, 32, 48, 64]
    
    for struct_info in all_structures:
        structure = struct_info['structure']
        
        try:
            # 原子番号と位置を取得
            numbers = np.array(structure['numbers']['__ndarray__'][2])
            positions = np.array(structure['positions']['__ndarray__'][2]).reshape(-1, 3)
            tags = np.array(structure['tags']['__ndarray__'][2])
            
            # 各レイヤーごとに処理（layer 4,3,2,1の順）
            row_data = []
            for layer in range(4, 0, -1):  # layer 4-1（逆順）
                layer_atoms = positions[tags == layer]
                layer_numbers = numbers[tags == layer]
                
                if len(layer_atoms) > 0:
                    # x座標でソート
                    sort_idx = np.argsort(layer_atoms[:, 0])
                    layer_atoms = layer_atoms[sort_idx]
                    layer_numbers = layer_numbers[sort_idx]
                    
                    # 16原子まで取得
                    for i in range(16):
                        if i < len(layer_numbers):
                            row_data.append(layer_numbers[i])
                        else:
                            row_data.append(0)  # 空の場合は0
                else:
                    # レイヤーに原子がない場合は0で埋める
                    row_data.extend([0] * 16)
            
            mosaic_data.append(row_data)
            y_offset += 1
            
        except Exception as e:
            print(f"Error processing structure {struct_info['iter_name']}-{struct_info['struct_key']}: {e}")
            continue
    
    # モザイク図を作成
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # 配列をnumpy配列に変換
    mosaic_array = np.array(mosaic_data)
    
    # カラーマップを作成
    color_map = np.zeros((mosaic_array.shape[0], mosaic_array.shape[1], 3))
    for i in range(mosaic_array.shape[0]):
        for j in range(mosaic_array.shape[1]):
            atom_num = mosaic_array[i, j]
            if atom_num == 28:  # Ni
                color_map[i, j] = [144/255, 238/255, 144/255]  # 薄緑
            elif atom_num == 78:  # Pt
                color_map[i, j] = [192/255, 192/255, 192/255]  # シルバー
            else:  # 空or他
                color_map[i, j] = [1, 1, 1]  # 白
    
    # モザイク図を描画（隙間なし）
    ax.imshow(color_map, aspect='auto', interpolation='nearest')
    
    # Y軸のラベル設定（iter番号のみ表示）
    # 各iterの開始位置と終了位置を計算
    iter_ranges = {}
    for i, struct_info in enumerate(all_structures):
        iter_name = struct_info['iter_name']
        if iter_name not in iter_ranges:
            iter_ranges[iter_name] = {'start': i, 'end': i}
        else:
            iter_ranges[iter_name]['end'] = i
    
    # Y軸の目盛りとラベルを設定
    y_ticks = []
    y_labels = []
    for iter_name in sorted(iter_ranges.keys(), key=lambda x: int(x.replace('iter', ''))):
        start = iter_ranges[iter_name]['start']
        end = iter_ranges[iter_name]['end']
        center = (start + end) / 2
        y_ticks.append(center)
        y_labels.append(iter_name)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    
    # iter間に境界線を追加
    for iter_name in sorted(iter_ranges.keys(), key=lambda x: int(x.replace('iter', ''))):
        if iter_ranges[iter_name]['end'] < len(all_structures) - 1:
            boundary = iter_ranges[iter_name]['end'] + 0.5
            ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=3)
    
    # レイヤー境界線とラベルを追加
    layer_positions = [15.5, 31.5, 47.5]  # 16原子ずつの境界
    for i, pos in enumerate(layer_positions):
        ax.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2)
    
    # レイヤーラベルを追加
    layer_labels = ['Layer 4', 'Layer 3', 'Layer 2', 'Layer 1']
    layer_centers = [7.5, 23.5, 39.5, 55.5]
    for i, (center, label) in enumerate(zip(layer_centers, layer_labels)):
        ax.text(center, -8, f'<-{label}->', 
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Atomic positions (-)', fontsize=12)
    ax.set_ylabel('Sample (-)', fontsize=12)
    ax.set_title(f'Structure Evolution (iter0 ~ iter{max_iter})', fontsize=14, fontweight='bold', pad=20)
    
    # X軸の設定
    ax.set_xlim(-0.5, 63.5)
    ax.set_xticks(range(0, 64, 8))  # 8原子おきに目盛り
    
    # 凡例を追加
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#C0C0C0', edgecolor='black', label='Pt'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='black', label='Ni')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
    
    plt.tight_layout()
    
    # 構造可視化を保存
    structure_output = output_path / f"structure_evolution_iter0-{max_iter}.png"
    plt.savefig(structure_output, dpi=300, bbox_inches='tight')
    print(f"Structure visualization saved: {structure_output}")
    plt.close()

def main():
    # コマンドライン引数を取得
    args = parse_args()
    
    print(f"Starting ORR catalyst data analysis (iter0 ~ iter{args.iter})...")
    print(f"Data directory: {args.base_path}")
    print(f"Output directory: {args.output_path}")
    
    # メイン分析の実行
    combined_df = analyze_orr_catalyst_data(
        max_iter=args.iter,
        base_path=args.base_path,
        output_path=args.output_path
    )
    
    if combined_df is not None and not combined_df.empty:
        output_path = Path(args.output_path)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Analyzed iterations: iter0 ~ iter{args.iter}")
        print(f"All figures saved to: {output_path}")
        print(f"Generated files:")
        for file in sorted(output_path.glob(f"*iter0-{args.iter}*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
    else:
        print("No data available for analysis.")
    
    print("Analysis script completed successfully!")

if __name__ == "__main__":
    main()