import json
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')  # バックエンドを非インタラクティブに設定
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

def load_json_data(file_path):
    """JSONファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_orr_catalyst_data():
    """ORR触媒データの分析とプロット作成"""
    
    # データファイルのパス設定
    base_path = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae/data")
    output_path = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae/result/figure")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # iter0, iter1, iter2のデータを読み込み
    all_data = {}
    for iter_num in [0, 1, 2, 3]:
        file_path = base_path / f"iter{iter_num}_calculation_result.json"
        if file_path.exists():
            data = load_json_data(file_path)
            all_data[f"iter{iter_num}"] = data
            print(f"iter{iter_num}: {len(data)} samples loaded")
        else:
            print(f"Warning: {file_path} not found")
    
    if not all_data:
        print("No data files found!")
        return
    
    # データをDataFrameに変換
    df_list = []
    for iter_name, data in all_data.items():
        df = pd.DataFrame(data)
        df['iter'] = iter_name
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} total samples")
    print(f"Iter distribution: {combined_df['iter'].value_counts()}")
    
    # カラーパレットの設定
    #colors = {'iter0': '#1f77b4', 'iter1': '#ff7f0e', 'iter2': '#2ca02c'}
    
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {f'iter{i}': mcolors.to_hex(tab10_colors[i]) for i in range(10)}
    
    # 全データのレンジを計算してヒストグラムの範囲を統一
    ni_fraction_clean = combined_df['ni_fraction'].dropna()
    overpotential_clean = combined_df['overpotential'].dropna()
    
    ni_range = (ni_fraction_clean.min(), ni_fraction_clean.max())
    overpot_range = (overpotential_clean.min(), overpotential_clean.max())
    
    # 1. Ni fraction のヒストグラム
    plt.figure(figsize=(10, 6))
    
    for iter_name in sorted(all_data.keys()):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        # NaNを除外してヒストグラムを作成
        ni_fraction_clean = iter_data['ni_fraction'].dropna()
        plt.hist(ni_fraction_clean, 
                bins=30, 
                range=ni_range,  #
                alpha=0.4, 
                label=iter_name, 
                color=colors.get(iter_name, 'gray'),
                edgecolor='black',
                linewidth=0.5)
    
    plt.xlabel('Ni Fraction', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Ni Fraction across Iterations', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    # ヒストグラムを保存
    hist_output = output_path / "ni_fraction_histogram.png"
    plt.savefig(hist_output, dpi=300, bbox_inches='tight')
    print(f"Histogram saved: {hist_output}")
    plt.close()
    
    # 2. Overpotential のヒストグラム
    plt.figure(figsize=(10, 6))
    
    for iter_name in sorted(all_data.keys()):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        # NaNを除外してヒストグラムを作成
        overpotential_clean = iter_data['overpotential'].dropna()
        plt.hist(overpotential_clean, 
                bins=30, 
                range=overpot_range,  # 範囲を統一
                alpha=0.4, 
                label=iter_name, 
                color=colors.get(iter_name, 'gray'),
                edgecolor='black',
                linewidth=0.5)
    
    plt.xlabel('Overpotential (V)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Overpotential across Iterations', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    # Overpotentialヒストグラムを保存
    overpot_hist_output = output_path / "overpotential_histogram.png"
    plt.savefig(overpot_hist_output, dpi=300, bbox_inches='tight')
    print(f"Overpotential histogram saved: {overpot_hist_output}")
    plt.close()
    
    # 3. Overpotential vs Ni fraction のscatter plot
    plt.figure(figsize=(10, 6))
    
    for iter_name in sorted(all_data.keys()):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        # NaNを除外
        valid_data = iter_data.dropna(subset=['ni_fraction', 'overpotential'])
        plt.scatter(valid_data['ni_fraction'], 
                   valid_data['overpotential'],
                   alpha=0.4,
                   label=iter_name,
                   color=colors.get(iter_name, 'gray'),
                   s=30,
                   edgecolors='black',
                   linewidths=0.3)
    
    plt.xlabel('Ni Fraction', fontsize=12)
    plt.ylabel('Overpotential (V)', fontsize=12)
    plt.title('Overpotential vs Ni Fraction across Iterations', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    # Scatter plotを保存
    scatter_output = output_path / "overpotential_vs_ni_fraction.png"
    plt.savefig(scatter_output, dpi=500, bbox_inches='tight')
    print(f"Scatter plot saved: {scatter_output}")
    plt.close()
    
    # 4. 統計サマリー
    print("\n=== Statistical Summary ===")
    for iter_name in sorted(all_data.keys()):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ni fractionのボックスプロット
    valid_ni_data = combined_df.dropna(subset=['ni_fraction'])
    if not valid_ni_data.empty:
        valid_ni_data.boxplot(column='ni_fraction', by='iter', ax=ax1)
        ax1.set_title('Ni Fraction Distribution by Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Ni Fraction')
    
    # Overpotentialのボックスプロット
    valid_overpot_data = combined_df.dropna(subset=['overpotential'])
    if not valid_overpot_data.empty:
        valid_overpot_data.boxplot(column='overpotential', by='iter', ax=ax2)
        ax2.set_title('Overpotential Distribution by Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Overpotential (V)')
    
    plt.suptitle('')  # デフォルトのタイトルを削除
    plt.tight_layout()
    
    # ボックスプロットを保存
    boxplot_output = output_path / "boxplots_comparison.png"
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
        
        summary_output = output_path / "summary_statistics.csv"
        summary_stats.to_csv(summary_output)
        print(f"Summary statistics saved: {summary_output}")
    except Exception as e:
        print(f"Warning: Could not save summary statistics: {e}")
    
    return combined_df

if __name__ == "__main__":
    print("Starting ORR catalyst data analysis...")
    
    # メイン分析の実行
    combined_df = analyze_orr_catalyst_data()
    
    if combined_df is not None and not combined_df.empty:
        output_path = Path("/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae/result/figure")
        
        print(f"\n=== Analysis Complete ===")
        print(f"All figures saved to: {output_path}")
        print(f"Generated files:")
        for file in sorted(output_path.glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
    else:
        print("No data available for analysis.")
    
    print("Analysis script completed successfully!")
