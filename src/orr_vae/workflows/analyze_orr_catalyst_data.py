#!/usr/bin/env python3
"""
ORR触媒データの分析とプロット作成スクリプト

主な機能:
- イテレーションごとのデータ分析
- ヒストグラム、散布図、ヴァイオリンプロットの作成
- 元素分率ヒートマップとPhase Diagramの作成
- 構造進化の可視化
"""

import json
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')  # バックエンドを非インタラクティブに設定
import matplotlib.pyplot as plt

# 全図のタイトル・軸ラベルを太字に統一
plt.rcParams.update({
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import argparse
from matplotlib.ticker import MultipleLocator
from ase.data.colors import jmol_colors, cpk_colors
from ase.data import atomic_numbers, chemical_symbols
from orr_vae.tool import ALLOY_ELEMENTS


# =============================================================================
# 設定とユーティリティ関数
# =============================================================================

SHOW_TITLES = False  # デフォルトはタイトル非表示（--title 指定時のみ表示）


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
    parser.add_argument('--title', action='store_true',
                        help='図タイトルを表示（デフォルトは非表示）')
    parser.add_argument('--palette', type=str, default='jmol',
                        choices=['jmol', 'cpk', 'tab10', 'set2'],
                        help='構造モザイク図で使用するカラーパレット (default: jmol)')
    return parser.parse_args()


def fraction_column(element: str) -> str:
    return f"{element.lower()}_fraction"


def get_available_fraction_elements(df: pd.DataFrame, include_pt: bool = True) -> list:
    """DataFrame内に存在する元素分率カラム（*_fraction）から元素名のリストを生成"""
    elements = []

    def _add(symbol: str):
        if (include_pt or symbol.lower() != 'pt') and symbol not in elements:
            elements.append(symbol)

    # 優先的に ALLOY_ELEMENTS の順序で追加
    for element in ALLOY_ELEMENTS:
        if fraction_column(element) in df.columns:
            _add(element)

    # 追加で任意の *_fraction カラムがあれば取り込む
    for col in df.columns:
        if not col.endswith('_fraction'):
            continue
        prefix = col[:-len('_fraction')]
        if not prefix:
            continue
        prefix = prefix.lower()
        symbol = prefix.capitalize()
        _add(symbol)

    # Pt を除外する指定のときに改めてフィルタ
    if not include_pt:
        elements = [elem for elem in elements if elem.lower() != 'pt']

    return elements


_DEFAULT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key().get(
    'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
)


def _fallback_rgb(symbol: str) -> tuple:
    color_hex = _DEFAULT_COLOR_CYCLE[hash(symbol) % len(_DEFAULT_COLOR_CYCLE)]
    return mcolors.to_rgb(color_hex)


def resolve_element_rgb(symbol: str) -> tuple:
    """元素シンボルからRGBタプル(0-1スケール)を取得"""
    try:
        z = atomic_numbers[symbol]
    except KeyError:
        return _fallback_rgb(symbol)

    if 0 <= z < len(jmol_colors):
        return tuple(jmol_colors[z][:3])
    return _fallback_rgb(symbol)


from collections import defaultdict

_CATEGORICAL_CACHE = defaultdict(dict)
_CATEGORICAL_ORDER = defaultdict(list)
_CATEGORICAL_COLORS = {}
_CATEGORICAL_INDEX = defaultdict(int)


def resolve_atomic_number_rgb(atomic_number: int, palette: str = "jmol") -> tuple:
    """原子番号からRGBタプル(0-1)を取得（未定義時は白）"""
    if palette == "cpk":
        source = cpk_colors
        if 0 < atomic_number < len(source):
            return tuple(source[atomic_number][:3])
    elif palette in {"tab10", "set2"}:
        symbol = chemical_symbols[atomic_number] if 0 < atomic_number < len(chemical_symbols) else f"Z{atomic_number}"
        cache = _CATEGORICAL_CACHE[palette]
        cmap_key = "Set2" if palette == "set2" else palette
        if palette not in _CATEGORICAL_COLORS:
            _CATEGORICAL_COLORS[palette] = list(plt.colormaps[cmap_key].colors)
            colors = _CATEGORICAL_COLORS[palette]
            for elem in ALLOY_ELEMENTS:
                if elem not in cache:
                    idx = _CATEGORICAL_INDEX[palette] % len(colors)
                    cache[elem] = tuple(colors[idx])
                    _CATEGORICAL_ORDER[palette].append(elem)
                    _CATEGORICAL_INDEX[palette] += 1
        if symbol not in cache:
            colors = _CATEGORICAL_COLORS[palette]
            idx = _CATEGORICAL_INDEX[palette] % len(colors)
            cache[symbol] = tuple(colors[idx])
            _CATEGORICAL_ORDER[palette].append(symbol)
            _CATEGORICAL_INDEX[palette] += 1
        return cache[symbol]
    else:  # default to jmol
        if 0 < atomic_number < len(jmol_colors):
            return tuple(jmol_colors[atomic_number][:3])
    return (1.0, 1.0, 1.0)


def load_json_data(file_path):
    """JSONファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_output_directories(output_path):
    """出力用のサブディレクトリを作成"""
    subdirs = {
        'histogram': output_path / "histogram",
        'scatter': output_path / "scatter_plot", 
        'violin': output_path / "violin_plot",
        'boxplot': output_path / "box_plot",
        'heatmap': output_path / "heatmap",
        'phase_diagram': output_path / "phase_diagram",
        'structure': output_path / "structure_visualization",
        'volcano_plot': output_path / "volcano_plot",
        'trend_plot': output_path / "trend_plot",
        'statistics': output_path / "statistics"
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def setup_color_palette(max_iter):
    """統一されたカラーパレットを設定（viridis, 0.1→1.0）。"""
    viridis_colors = plt.cm.viridis(np.linspace(0.1, 1.0, max_iter + 1))
    return {f'iter{i}': mcolors.to_hex(viridis_colors[i]) for i in range(max_iter + 1)}


def get_iter_order(df):
    """イテレーションの並び順を統一"""
    return sorted(df["iter"].unique(), key=lambda s: int(s.replace("iter", "")))


# =============================================================================
# データ読み込みと前処理
# =============================================================================

def load_iteration_data(base_path, max_iter):
    """イテレーションデータを読み込む"""
    all_data = {}
    iter_range = list(range(max_iter + 1))
    
    print(f"=== データ読み込み (iter0 - iter{max_iter}) ===")
    
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
    return all_data


def prepare_dataframe(all_data):
    """データをDataFrameに変換し前処理を行う"""
    df_list = []
    for iter_name, data in all_data.items():
        df = pd.DataFrame(data)
        df['iter'] = iter_name
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} total samples")
    print(f"Iter distribution: {combined_df['iter'].value_counts().sort_index()}")
    
    return combined_df


def calculate_data_ranges(combined_df):
    """データの範囲を計算"""
    alloy_formation_clean = combined_df['E_alloy_formation'].dropna()
    overpotential_clean = combined_df['overpotential'].dropna()
    
    alloy_formation_range = (alloy_formation_clean.min(), alloy_formation_clean.max())
    overpot_range = (overpotential_clean.min(), overpotential_clean.max())
    
    print(f"Data ranges - Alloy Formation Energy: {alloy_formation_range[0]:.3f} ~ {alloy_formation_range[1]:.3f} eV/atom")
    print(f"Data ranges - Overpotential: {overpot_range[0]:.3f} ~ {overpot_range[1]:.3f} V")
    
    return alloy_formation_range, overpot_range


# =============================================================================
# 基本プロット作成関数
# =============================================================================

def create_histogram(combined_df, all_data, colors, data_range, 
                    column, xlabel, title, filename, max_iter):
    """ヒストグラムを作成"""
    plt.figure(figsize=(12, 10))
    
    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        clean_data = iter_data[column].dropna()
        
        plt.hist(clean_data, 
                bins=30, 
                range=data_range,
                alpha=0.5, 
                label=f'{iter_name}', 
                color=colors.get(iter_name, 'gray'),
                edgecolor='black',
                linewidth=0.5)
    
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    if SHOW_TITLES:
        plt.title(title, fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_scatter_plot(combined_df, all_data, colors, x_col, y_col,
                       xlabel, ylabel, title, filename, max_iter):
    """散布図を作成"""
    plt.figure(figsize=(12, 12))
    
    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        valid_data = iter_data.dropna(subset=[x_col, y_col])
        if len(valid_data) > 0:
            plt.scatter(
                valid_data[x_col],
                valid_data[y_col],
                alpha=0.85,
                label=f'{iter_name}',
                color=colors.get(iter_name, 'gray'),
                s=100,
                edgecolors='black',
                linewidths=0.2,
            )
    
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if SHOW_TITLES:
        plt.title(title, fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()


def print_statistical_summary(combined_df, all_data, max_iter):
    """統計サマリーを出力"""
    print(f"\n=== Statistical Summary (iter0 - iter{max_iter}) ===")
    available_elements = get_available_fraction_elements(combined_df)

    for iter_name in sorted(all_data.keys(), key=lambda x: int(x.replace('iter', ''))):
        iter_data = combined_df[combined_df['iter'] == iter_name]
        overpotential_clean = iter_data['overpotential'].dropna()
        alloy_formation_clean = iter_data['E_alloy_formation'].dropna()
        
        print(f"\n{iter_name}:")
        print(f"  Total samples: {len(iter_data)}")
        print(f"  Alloy formation samples: {len(alloy_formation_clean)}")
        print(f"  Overpotential samples: {len(overpotential_clean)}")
        
        if len(alloy_formation_clean) > 0:
            print(f"  Alloy formation - Mean: {alloy_formation_clean.mean():.3f}, Std: {alloy_formation_clean.std():.3f}")
        
        if len(overpotential_clean) > 0:
            print(f"  Overpotential - Mean: {overpotential_clean.mean():.3f}, Std: {overpotential_clean.std():.3f}")
            print(f"  Min overpotential: {overpotential_clean.min():.3f}")
            
            # 最小overpotentialのAlloy formation energyを取得
            min_idx = iter_data['overpotential'].idxmin()
            if pd.notna(iter_data.loc[min_idx, 'E_alloy_formation']):
                print(f"    (Alloy formation energy: {iter_data.loc[min_idx, 'E_alloy_formation']:.3f} eV/atom)")

        for element in available_elements:
            col = fraction_column(element)
            if col not in iter_data.columns:
                continue
            element_data = iter_data[col].dropna()
            if len(element_data) == 0:
                continue
            print(
                f"  {element} fraction - Count: {len(element_data)}, "
                f"Mean: {element_data.mean():.3f}, Std: {element_data.std():.3f}"
            )


# =============================================================================
# ヴァイオリンプロット作成関数
# =============================================================================
def create_violin_plot(data, y_col, ylabel, title, filename):
    """ヴァイオリンプロットを作成"""
    order = get_iter_order(data)
    # 一貫したiter色（viridis 0.1→1.0）を適用
    try:
        max_iter = max(int(s.replace("iter", "")) for s in order)
    except Exception:
        max_iter = len(order) - 1
    color_map = setup_color_palette(max_iter)
    palette = [color_map[it] for it in order]

    fig, ax = plt.subplots(figsize=(12, 8))

    # 全データ点を左側に配置
    sns.stripplot(data=data, x="iter", y=y_col, order=order,
                  dodge=False, jitter=0.01, size=5, color="k", 
                  alpha=0.6, ax=ax)
    
    # 各点の位置を左側にずらす
    for collection in ax.collections:
        if hasattr(collection, '_offsets'):
            offsets = collection.get_offsets()
            offsets[:, 0] -= 0.4
            collection.set_offsets(offsets)

    # ヴァイオリン
    sns.violinplot(data=data, x="iter", y=y_col, hue="iter",
                   order=order, hue_order=order, palette=palette,
                   linewidth=1, inner=None, width=0.5, legend=False, 
                   alpha=0.8, ax=ax)
    
    # 箱ひげ
    sns.boxplot(data=data, x="iter", y=y_col, order=order, width=0.05,
                showcaps=True, boxprops={"facecolor": "none", "zorder": 3},
                whiskerprops={"linewidth": 1.5},
                medianprops={"color": "black", "linewidth": 1.5}, ax=ax)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.margins(y=0.05)

    # 凡例を追加
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[i], 
                                   alpha=0.8, label=iter_name)
                      for i, iter_name in enumerate(order)]
    ax.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5), fontsize=10)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_combined_violin_plot(valid_alloy_data, valid_overpot_data, 
                               filename, max_iter):
    """統合されたヴァイオリンプロットを作成"""
    order = get_iter_order(valid_alloy_data)
    color_map = setup_color_palette(max_iter)
    palette = [color_map[it] for it in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    # 左: Alloy Formation
    _create_violin_subplot(axes[0], valid_alloy_data, "E_alloy_formation", 
                          "Alloy Formation Energy", "Formation Energy (eV/atom)", 
                          order, palette)

    # 右: Overpotential  
    _create_violin_subplot(axes[1], valid_overpot_data, "overpotential",
                          "Overpotential", "Overpotential (V)", 
                          order, palette)

    if SHOW_TITLES:
        fig.suptitle(f"Distribution Analysis (iter0 - iter{max_iter})", 
                    fontsize=28, fontweight='bold')
    
    # 凡例を右側に追加
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[i], 
                                   alpha=0.8, label=iter_name)
                      for i, iter_name in enumerate(order)]
    fig.legend(handles=legend_elements, loc='center left', 
               bbox_to_anchor=(1, 0.5), fontsize=16)
    
    fig.tight_layout(rect=[0, 0.03, 0.99, 0.95])
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _create_violin_subplot(ax, data, y_col, title, ylabel, order, palette):
    """ヴァイオリンプロットのサブプロットを作成"""
    # データ点
    sns.stripplot(data=data, x="iter", y=y_col, order=order, 
                  color="k", size=5, jitter=0.01, alpha=0.6, ax=ax)
    
    # 点を左側にずらす
    for collection in ax.collections:
        if hasattr(collection, '_offsets'):
            offsets = collection.get_offsets()
            offsets[:, 0] -= 0.4
            collection.set_offsets(offsets)
    
    # ヴァイオリン
    sns.violinplot(data=data, x="iter", y=y_col, hue="iter",
                   order=order, hue_order=order, palette=palette, 
                   inner=None, width=0.5, legend=False, alpha=0.8, ax=ax)
    
    # 箱ひげ
    sns.boxplot(data=data, x="iter", y=y_col, order=order, width=0.05,
                boxprops={"facecolor": "none", "zorder": 3},
                whiskerprops={"linewidth": 1.5},
                medianprops={"color": "black", "linewidth": 1.5}, ax=ax)
    
    if SHOW_TITLES:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.grid(True, alpha=0.3)


def create_boxplots(valid_alloy_data, valid_overpot_data, colors, 
                   filename, max_iter):
    """従来の箱ひげ図を作成"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    if not valid_alloy_data.empty:
        iter_order = sorted(valid_alloy_data['iter'].unique(), 
                           key=lambda x: int(x.replace('iter', '')))
        palette = [colors.get(iter_name, 'gray') for iter_name in iter_order]
        
        sns.boxplot(data=valid_alloy_data, x='iter', y='E_alloy_formation', 
                    ax=ax1, order=iter_order, palette=palette, 
                    hue='iter', legend=False)
        if SHOW_TITLES:
            ax1.set_title(f'Alloy Formation Energy Distribution (iter0 - iter{max_iter})', fontsize=20)
        ax1.set_xlabel('Iteration', fontsize=20)
        ax1.set_ylabel('Alloy Formation Energy (eV/atom)', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid(True, alpha=0.3)
    
    if not valid_overpot_data.empty:
        iter_order = sorted(valid_overpot_data['iter'].unique(), 
                           key=lambda x: int(x.replace('iter', '')))
        palette = [colors.get(iter_name, 'gray') for iter_name in iter_order]
        
        sns.boxplot(data=valid_overpot_data, x='iter', y='overpotential', 
                    ax=ax2, order=iter_order, palette=palette, 
                    hue='iter', legend=False)
        if SHOW_TITLES:
            ax2.set_title(f'Overpotential Distribution (iter0 - iter{max_iter})', fontsize=20)
        ax2.set_xlabel('Iteration', fontsize=20)
        ax2.set_ylabel('Overpotential (V)', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_summary_statistics(combined_df, filename):
    """統計サマリーをCSVに保存"""
    try:
        agg_dict = {
            'E_alloy_formation': ['count', 'mean', 'std', 'min', 'max'],
            'overpotential': ['count', 'mean', 'std', 'min', 'max'],
            'limiting_potential': ['count', 'mean', 'std']
        }

        for element in get_available_fraction_elements(combined_df):
            col = fraction_column(element)
            if col in combined_df.columns:
                agg_dict[col] = ['count', 'mean', 'std', 'min', 'max']

        summary_stats = combined_df.groupby('iter').agg(agg_dict).round(4)
        
        summary_stats.to_csv(filename)
        print(f"Summary statistics saved: {filename}")
    except Exception as e:
        print(f"Warning: Could not save summary statistics: {e}")


# =============================================================================
# メイン分析関数
# =============================================================================

def analyze_orr_catalyst_data(max_iter=4, base_path=None, output_path=None, palette="jmol"):
    """ORR触媒データの分析とプロット作成のメイン関数"""
    
    # パス設定
    base_path = Path(base_path) if base_path else Path(__file__).parent / "data"
    output_path = Path(output_path) if output_path else Path(__file__).parent / "result" / "figure"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリを作成
    subdirs = create_output_directories(output_path)
    
    print(f"=== ORR触媒データ分析 (iter0 - iter{max_iter}) ===")
    print(f"データディレクトリ: {base_path}")
    print(f"出力ディレクトリ: {output_path}")
    
    # データ読み込み
    all_data = load_iteration_data(base_path, max_iter)
    if not all_data:
        return None
    
    # データ前処理
    combined_df = prepare_dataframe(all_data)
    colors = setup_color_palette(max_iter)
    alloy_formation_range, overpot_range = calculate_data_ranges(combined_df)
    
    # 基本プロット作成
    _create_basic_plots(combined_df, all_data, colors, alloy_formation_range, 
                       overpot_range, subdirs, max_iter)
    
    # 統計サマリー出力
    print_statistical_summary(combined_df, all_data, max_iter)
    
    # ヴァイオリンプロット作成
    _create_violin_plots(combined_df, subdirs, max_iter)
    
    # 箱ひげ図作成
    _create_comparison_plots(combined_df, colors, subdirs, max_iter)
    
    # 統計データ保存
    save_summary_statistics(combined_df, 
                           subdirs['statistics'] / f"summary_statistics_iter0-{max_iter}.csv")
    
    # 構造データ処理
    _process_structure_data(base_path, subdirs, max_iter, palette)
    
    # 高度な分析
    create_element_fraction_heatmaps(combined_df, subdirs, max_iter)
    create_phase_diagram(combined_df, subdirs, max_iter)
    
    # ORRプロット作成
    create_all_orr_plots(combined_df, subdirs, max_iter)
    
    return combined_df


def _create_basic_plots(combined_df, all_data, colors, alloy_formation_range, 
                       overpot_range, subdirs, max_iter):
    """基本的なプロットを作成"""
    # ヒストグラム
    create_histogram(combined_df, all_data, colors, alloy_formation_range,
                    'E_alloy_formation', 'Alloy Formation Energy (eV/atom)',
                    f'Distribution of Alloy Formation Energy (iter0 - iter{max_iter})',
                    subdirs['histogram'] / f"alloy_formation_histogram_iter0-{max_iter}.png",
                    max_iter)
    
    create_histogram(combined_df, all_data, colors, overpot_range,
                    'overpotential', 'Overpotential (V)',
                    f'Distribution of Overpotential (iter0 - iter{max_iter})',
                    subdirs['histogram'] / f"overpotential_histogram_iter0-{max_iter}.png",
                    max_iter)
    
    # 散布図
    create_scatter_plot(combined_df, all_data, colors, 
                       'E_alloy_formation', 'overpotential',
                       'Alloy Formation Energy (eV/atom)', 'Overpotential (V)',
                       f'Overpotential vs Alloy Formation Energy (iter0 - iter{max_iter})',
                       subdirs['scatter'] / f"overpotential_vs_alloy_formation_iter0-{max_iter}.png",
                       max_iter)


def _create_violin_plots(combined_df, subdirs, max_iter):
    """ヴァイオリンプロットを作成"""
    valid_alloy_data = combined_df.dropna(subset=["E_alloy_formation"])
    valid_overpot_data = combined_df.dropna(subset=["overpotential"])
    
    if not valid_alloy_data.empty:
        create_violin_plot(valid_alloy_data, "E_alloy_formation",
                          "Alloy Formation Energy (eV/atom)",
                          f"Alloy Formation Energy Distribution (iter0 - iter{max_iter})",
                          subdirs['violin'] / f"violin_alloy_formation_iter0-{max_iter}.png")

    if not valid_overpot_data.empty:
        create_violin_plot(valid_overpot_data, "overpotential",
                          "Overpotential (V)",
                          f"Overpotential Distribution (iter0 - iter{max_iter})",
                          subdirs['violin'] / f"violin_overpotential_iter0-{max_iter}.png")

    if not valid_alloy_data.empty and not valid_overpot_data.empty:
        create_combined_violin_plot(valid_alloy_data, valid_overpot_data,
                                   subdirs['violin'] / f"violin_combined_iter0-{max_iter}.png",
                                   max_iter)


def _create_comparison_plots(combined_df, colors, subdirs, max_iter):
    """比較用プロットを作成"""
    valid_alloy_data = combined_df.dropna(subset=["E_alloy_formation"])
    valid_overpot_data = combined_df.dropna(subset=["overpotential"])
    
    create_boxplots(valid_alloy_data, valid_overpot_data, colors,
                   subdirs['boxplot'] / f"boxplots_comparison_iter0-{max_iter}.png",
                   max_iter)


def _process_structure_data(base_path, subdirs, max_iter, palette: str):
    """構造データを処理"""
    structure_data = {}
    for iter_num in range(max_iter + 1):
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
        create_structure_visualization(structure_data, subdirs, max_iter, palette)


# =============================================================================
# 構造可視化関数
# =============================================================================

def create_structure_visualization(structure_data, subdirs, max_iter, palette: str):
    """構造の可視化を作成"""
    print(f"\n=== Creating Structure Visualization ===")
    
    # 全ての構造を取得
    all_structures = _extract_all_structures(structure_data)
    if not all_structures:
        print("No structures found for visualization")
        return
    
    # 構造をソート
    all_structures.sort(key=lambda x: (x['iter'], x['struct_num']))
    
    # モザイク用データ作成
    mosaic_data = _create_mosaic_data(all_structures)
    
    # モザイク図作成
    _create_structure_mosaic(mosaic_data, all_structures, subdirs, max_iter, palette)

    # 追加: 層ごとの元素原子数推移プロット
    _create_element_layer_count_plots(structure_data, subdirs, max_iter)


def _extract_all_structures(structure_data):
    """全ての構造を抽出"""
    all_structures = []
    for iter_name, structures in structure_data.items():
        iter_num = int(iter_name.replace('iter', ''))
        for struct_key, structure in structures.items():
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
                continue
    return all_structures


def _create_mosaic_data(all_structures):
    """モザイク用の配列を作成"""
    mosaic_data = []
    
    for struct_info in all_structures:
        structure = struct_info['structure']
        
        try:
            numbers = np.array(structure['numbers']['__ndarray__'][2])
            positions = np.array(structure['positions']['__ndarray__'][2]).reshape(-1, 3)
            tags = np.array(structure['tags']['__ndarray__'][2])
            
            # 各レイヤーごとに処理
            row_data = []
            for layer in range(4, 0, -1):
                layer_atoms = positions[tags == layer]
                layer_numbers = numbers[tags == layer]
                
                if len(layer_atoms) > 0:
                    sort_idx = np.argsort(layer_atoms[:, 0])
                    layer_numbers = layer_numbers[sort_idx]
                    
                    for i in range(16):
                        if i < len(layer_numbers):
                            row_data.append(layer_numbers[i])
                        else:
                            row_data.append(0)
                else:
                    row_data.extend([0] * 16)
            
            mosaic_data.append(row_data)
            
        except Exception as e:
            print(f"Error processing structure {struct_info['iter_name']}-{struct_info['struct_key']}: {e}")
            continue
    
    return mosaic_data


def _create_structure_mosaic(mosaic_data, all_structures, subdirs, max_iter, palette: str):
    """モザイク図を作成"""
    # 一回り小さくし、文字は大きく太字に
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # カラーマップを作成
    mosaic_array = np.array(mosaic_data, dtype=int)
    color_map = np.zeros((mosaic_array.shape[0], mosaic_array.shape[1], 3))

    for i in range(mosaic_array.shape[0]):
        for j in range(mosaic_array.shape[1]):
            atom_num = mosaic_array[i, j]
            color_map[i, j] = resolve_atomic_number_rgb(atom_num, palette)
    
    # モザイク図を描画
    ax.imshow(color_map, aspect='auto', interpolation='nearest')
    
    # Y軸設定
    _setup_y_axis(ax, all_structures)
    
    # レイヤー境界線とラベル
    _setup_layer_boundaries(ax)
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Atomic position index', fontsize=20, fontweight='bold')
    ax.set_ylabel('Iteration step', fontsize=20, fontweight='bold')
    if SHOW_TITLES:
        ax.set_title(
            f'Structural evolution (iter0 - iter{max_iter})',
            fontsize=20, fontweight='bold', pad=12,
        )
    
    # X軸設定（主要目盛は8刻み、補助目盛を1刻みで追加）
    ax.set_xlim(-0.5, 63.5)
    ax.set_xticks(range(0, 64, 8))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', length=3, width=0.8)
    
    # 凡例
    unique_atomic_numbers = [int(num) for num in np.unique(mosaic_array) if num != 0]
    symbol_by_z = {
        z: chemical_symbols[z] if 0 < z < len(chemical_symbols) else f'Z{z}'
        for z in unique_atomic_numbers
    }

    ordered_symbols = []
    for element in ALLOY_ELEMENTS:
        for z, symbol in symbol_by_z.items():
            if symbol == element and (symbol, z) not in ordered_symbols:
                ordered_symbols.append((symbol, z))
    for z, symbol in symbol_by_z.items():
        if all(symbol != existing[0] for existing in ordered_symbols):
            ordered_symbols.append((symbol, z))

    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=mcolors.to_hex(resolve_atomic_number_rgb(z, palette)),
            edgecolor='black',
            label=symbol
        )
        for symbol, z in ordered_symbols
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=20,
    )

    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # レイアウトを整えて凡例用の余白も確保
    plt.tight_layout(rect=(0, 0, 0.88, 0.96))
    
    # 保存
    structure_output = subdirs['structure'] / f"structure_evolution_iter0-{max_iter}.png"
    plt.savefig(structure_output, dpi=300, bbox_inches='tight')
    print(f"Structure visualization saved: {structure_output}")
    plt.close()


def _create_element_layer_count_plots(structure_data, subdirs, max_iter):
    """各層に存在する元素原子数の推移を出力"""
    elements = []
    encountered_atomic_numbers = set()

    for element in ALLOY_ELEMENTS:
        try:
            z = atomic_numbers[element]
        except KeyError:
            continue
        elements.append((element, z))
        encountered_atomic_numbers.add(z)

    # 構造データ内に存在するその他の元素も追加
    for structures in structure_data.values():
        for structure in structures.values():
            try:
                numbers = np.array(structure['numbers']['__ndarray__'][2]).astype(int)
            except Exception:
                continue
            for z in np.unique(numbers):
                if z == 0 or z in encountered_atomic_numbers:
                    continue
                symbol = chemical_symbols[z] if 0 < z < len(chemical_symbols) else f'Z{z}'
                elements.append((symbol, z))
                encountered_atomic_numbers.add(z)

    for label, atomic_number in elements:
        try:
            counts_by_iter = {}

            for iter_name, structures in structure_data.items():
                try:
                    iter_num = int(iter_name.replace('iter', ''))
                except Exception:
                    continue

                counts_by_iter.setdefault(iter_num, {1: [], 2: [], 3: [], 4: []})

                for structure in structures.values():
                    try:
                        numbers = np.array(structure['numbers']['__ndarray__'][2])
                        positions = np.array(structure['positions']['__ndarray__'][2]).reshape(-1, 3)
                        tags = np.array(structure['tags']['__ndarray__'][2])

                        unique_tags = np.unique(tags)
                        tag_z = []
                        for t in unique_tags:
                            mask = tags == t
                            if np.any(mask):
                                tag_z.append((t, float(np.mean(positions[mask, 2]))))
                        tag_z.sort(key=lambda x: x[1], reverse=True)
                        top4 = [tz[0] for tz in tag_z[:4]]

                        element_mask = (numbers == atomic_number)
                        for li in range(1, 5):
                            if li - 1 < len(top4):
                                t = top4[li - 1]
                                layer_count = int(np.sum(element_mask & (tags == t)))
                            else:
                                layer_count = 0
                            counts_by_iter[iter_num][li].append(layer_count)
                    except Exception:
                        continue

            if not counts_by_iter:
                print(f"No structure data available for {label} layer-count plot.")
                continue

            has_element_data = any(
                any(val > 0 for val in counts_by_iter[it][li])
                for it in counts_by_iter
                for li in range(1, 5)
            )

            if not has_element_data:
                print(f"No {label} atoms detected for layer-count plot.")
                continue

            iters = sorted(counts_by_iter.keys())
            layer_means = {li: [] for li in range(1, 5)}
            max_value = 0

            for it in iters:
                for li in range(1, 5):
                    vals = counts_by_iter[it][li]
                    mean_val = float(np.mean(vals)) if len(vals) > 0 else np.nan
                    layer_means[li].append(mean_val)
                    if len(vals) > 0:
                        max_value = max(max_value, max(vals))

            layer_colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red'}

            fig, ax = plt.subplots(figsize=(12, 10))
            for li in range(1, 5):
                ax.plot(
                    iters,
                    layer_means[li],
                    marker='o',
                    linewidth=2.5,
                    markersize=7,
                    color=layer_colors[li],
                    label=f'Layer {li}' + (' (top layer)' if li == 1 else ''),
                )

            ax.set_xlabel('Iteration', fontsize=20, fontweight='bold')
            ax.set_ylabel(f'Number of {label} atoms', fontsize=20, fontweight='bold')
            if SHOW_TITLES:
                ax.set_title(
                    f'{label} Atom Count per Layer vs Iteration (iter0 - iter{max_iter})',
                    fontsize=20,
                    fontweight='bold',
                    pad=12,
                )

            ax.set_xticks(range(min(iters), max(iters) + 1))
            ax.xaxis.set_major_locator(MultipleLocator(1))

            y_upper = max_value + 2
            if y_upper < 4:
                y_upper = 4
            ax.set_ylim(0, y_upper)
            y_step = max(1, int(np.ceil(y_upper / 8)))
            ax.yaxis.set_major_locator(MultipleLocator(y_step))

            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.grid(True, alpha=0.3)

            ax.legend(fontsize=16, loc='upper left', framealpha=0.9)

            fig.tight_layout()
            out_path = subdirs['structure'] / (
                f"{label.lower()}_layer_count_vs_iteration_iter0-{max_iter}.png"
            )
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"{label} layer-count plot saved: {out_path}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating {label} layer-count plot: {e}")


def _setup_y_axis(ax, all_structures):
    """Y軸の設定"""
    iter_ranges = {}
    for i, struct_info in enumerate(all_structures):
        iter_name = struct_info['iter_name']
        if iter_name not in iter_ranges:
            iter_ranges[iter_name] = {'start': i, 'end': i}
        else:
            iter_ranges[iter_name]['end'] = i
    
    # Y軸の目盛り（目盛線）は消し、iter ラベルは残す
    y_ticks = []
    y_labels = []
    for iter_name in sorted(iter_ranges.keys(), key=lambda x: int(x.replace('iter', ''))):
        start = iter_ranges[iter_name]['start']
        end = iter_ranges[iter_name]['end']
        center = (start + end) / 2
        y_ticks.append(center)
        y_labels.append(iter_name)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=20, fontweight='bold')
    # 目盛り（tick marks）のみ非表示にする（ラベルは表示）
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    
    # iter間の境界線
    for iter_name in sorted(iter_ranges.keys(), key=lambda x: int(x.replace('iter', ''))):
        if iter_ranges[iter_name]['end'] < len(all_structures) - 1:
            boundary = iter_ranges[iter_name]['end'] + 0.5
            ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=3)


def _setup_layer_boundaries(ax):
    """レイヤー境界線とラベルを設定"""
    layer_positions = [15.5, 31.5, 47.5]
    for pos in layer_positions:
        ax.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2)
    # レイヤー名の表記は重なり回避のため省略（縦線のみ表示）


# =============================================================================
# ORR Volcano Plot と Trend Plot 関数
# =============================================================================

def calculate_dG_values(combined_df):
    """JSONデータからdG値を計算
    
    diffG_U0からdG値を抽出 (https://doi.org/10.1038/s41524-019-0210-3)：
    dG1 = dG_OOH - 4 * 1.23 eV    
    dG2 = dG_O   - dG_OOH         
    dG3 = dG_OH  - dG_O           
    dG4 = - dG_OH                
    
    実際のdG値への変換：
    dG_OH = -dG4
    dG_OOH = dG1 + 4 * 1.23
    dG_O = dG_OH - dG3
    """
    print(f"\n=== Calculating dG values ===")
    
    # diffG_U0からdG値を抽出
    
    def extract_dG_values(row):
        try:
            if 'diffG_U0' in row and row['diffG_U0'] is not None:
                diffG_U0 = row['diffG_U0']
                if isinstance(diffG_U0, list) and len(diffG_U0) >= 4:
                    dG1 = diffG_U0[0]  # dG1 = dG_OOH - 4 * 1.23 eV
                    dG2 = diffG_U0[1]  # dG2 = dG_O - dG_OOH
                    dG3 = diffG_U0[2]  # dG3 = dG_OH - dG_O
                    dG4 = diffG_U0[3]  # dG4 = - dG_OH
                    
                    # 実際のdG値に変換
                    # dG_OH = -dG4
                    dG_OH_actual = -dG4
                    # dG_OOH = dG1 + 4 * 1.23
                    dG_OOH_actual = dG1 + 4 * 1.23
                    # dG_O = dG_OH - dG3
                    dG_O_actual = dG_OH_actual - dG3
                    
                    return pd.Series({
                        'dG_OOH': dG_OOH_actual,
                        'dG_O': dG_O_actual,
                        'dG_OH': dG_OH_actual
                    })
            return pd.Series({'dG_OOH': np.nan, 'dG_O': np.nan, 'dG_OH': np.nan})
        except Exception as e:
            print(f"Error calculating dG values: {e}")
            return pd.Series({'dG_OOH': np.nan, 'dG_O': np.nan, 'dG_OH': np.nan})
    
    # dG値を計算
    dG_values = combined_df.apply(extract_dG_values, axis=1)
    combined_df = pd.concat([combined_df, dG_values], axis=1)
    
    # 有効性を確認
    valid_dG_data = combined_df.dropna(subset=['dG_OH', 'dG_O', 'dG_OOH', 'limiting_potential'])
    print(f"Valid samples with dG values: {len(valid_dG_data)}")
    
    if len(valid_dG_data) > 0:
        print(f"dG_OH range: {valid_dG_data['dG_OH'].min():.3f} ~ {valid_dG_data['dG_OH'].max():.3f} eV")
        print(f"dG_O range: {valid_dG_data['dG_O'].min():.3f} ~ {valid_dG_data['dG_O'].max():.3f} eV")
        print(f"dG_OOH range: {valid_dG_data['dG_OOH'].min():.3f} ~ {valid_dG_data['dG_OOH'].max():.3f} eV")
        print(f"Limiting potential range: {valid_dG_data['limiting_potential'].min():.3f} ~ {valid_dG_data['limiting_potential'].max():.3f} V")
        
        # 計算例を表示（最初の有効なサンプル）
        first_valid = valid_dG_data.iloc[0]
        if 'diffG_U0' in first_valid and first_valid['diffG_U0'] is not None:
            diffG_U0 = first_valid['diffG_U0']
            print(f"\nExample calculation (first valid sample):")
            print(f"  diffG_U0: {diffG_U0}")
            print(f"  dG1 (dG_OOH - 4*1.23): {diffG_U0[0]:.3f}")
            print(f"  dG2 (dG_O - dG_OOH): {diffG_U0[1]:.3f}")
            print(f"  dG3 (dG_OH - dG_O): {diffG_U0[2]:.3f}")
            print(f"  dG4 (-dG_OH): {diffG_U0[3]:.3f}")
            print(f"  → dG_OH = -dG4 = {-diffG_U0[3]:.3f} eV")
            print(f"  → dG_OOH = dG1 + 4*1.23 = {diffG_U0[0] + 4*1.23:.3f} eV")
            print(f"  → dG_O = dG_OH - dG3 = {-diffG_U0[3] - diffG_U0[2]:.3f} eV")
    
    return combined_df


def create_orr_volcano_plot(data, x_col, colors, output_file, max_iter, color_type="iter"):
    """ORR Volcano Plot を作成（余白最適化版）。

    ポイント:
    - データに合わせた動的な x/y 範囲（0.5刻みに丸め）。
    - 等尺性は解除（set_aspect('auto')）して不要な余白を削減。
    - bbox_inches='tight' + pad_inches=0.05 で最終的な余白を圧縮。
    - 目盛は従来どおり 0.5 間隔、理論線は全表示範囲に合わせて描画。
    """
    print(f"\n=== Creating ORR Volcano Plot: {x_col} vs Limiting Potential ===")

    # 有効なデータのみを使用
    valid_data = data.dropna(subset=[x_col, 'limiting_potential'])
    if len(valid_data) == 0:
        print("No valid data found for volcano plot!")
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    if color_type == "iter":
        # イテレーションごとに色分け
        iter_order = get_iter_order(valid_data)
        color_map = setup_color_palette(max_iter)
        for it in iter_order:
            d = valid_data[valid_data['iter'] == it]
            if len(d) > 0:
                ax.scatter(d[x_col], d['limiting_potential'],
                           c=color_map.get(it, 'gray'), s=100, alpha=0.85,
                           edgecolors='black', linewidths=0.2, label=f'{it}')
        legend_title = 'Iterations'
    else:
        # 元素含有量でヒートマップ（カラーバーは外側）
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fraction_col = f'{color_type}_fraction'
        if fraction_col not in valid_data.columns:
            print(f"  Warning: {fraction_col} not found. Skipping volcano plot colored by {color_type}.")
            return
        sc = ax.scatter(valid_data[x_col], valid_data['limiting_potential'],
                        c=valid_data[fraction_col], cmap='RdYlBu_r',
                        s=100, alpha=0.85, edgecolors='black', linewidths=0.2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.15)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(f'{color_type.capitalize()} Fraction', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=20)
        legend_title = f'{color_type.capitalize()} Fraction'

    # 動的な軸範囲（0.5 刻みに丸め）
    def _nice_bounds(lo, hi, step=0.5, pad_frac=0.05):
        import math
        span = max(hi - lo, 1e-6)
        pad = span * pad_frac
        lo2 = lo - pad
        hi2 = hi + pad
        lo_round = math.floor(lo2 / step) * step
        hi_round = math.ceil(hi2 / step) * step
        return lo_round, hi_round

    x_min, x_max = valid_data[x_col].min(), valid_data[x_col].max()
    # 仮の x 範囲（後で最終確定）。dG_OH の場合は固定レンジにする
    forced_oh = (x_col == 'dG_OH')
    x_range_pre = (-0.5, 1.5) if forced_oh else _nice_bounds(x_min, x_max, step=0.5, pad_frac=0.05)

    # 理論線計算用 x（最終 x_range に合わせて後で再設定するため一旦準備）
    x_vals = np.linspace(x_range_pre[0], x_range_pre[1], 200)
    ax.axhline(y=1.23, color='k', linestyle='solid', linewidth=2, alpha=0.8, label='Ideal (1.23 V)')

    if x_col == 'dG_OH':
        y1 = -x_vals + 1.72
        y2 = x_vals
        ax.plot(
            x_vals, y1,
            color='red', linestyle='dotted', alpha=0.8, linewidth=2,
            label=r'$\mathrm{O_2\ \rightarrow\ OOH^{*}}$'
        )
        ax.plot(
            x_vals, y2,
            color='blue', linestyle='dashed', alpha=0.8, linewidth=2,
            label=r'$\mathrm{OH^{*}\ \rightarrow\ H_2O}$'
        )
    else:  # dG_O
        y1 = -0.5 * x_vals + 1.72
        y2 = 0.5 * x_vals
        ax.plot(
            x_vals, y1,
            color='red', linestyle='dotted', alpha=0.8, linewidth=2,
            label=r'$\mathrm{OH^{*}\ \rightarrow\ H_2O}$'
        )
        ax.plot(
            x_vals, y2,
            color='blue', linestyle='dashed', alpha=0.8, linewidth=2,
            label=r'$\mathrm{O_2\ \rightarrow\ OOH^{*}}$'
        )

    # y 範囲もデータ + 理論線を含むように決定（dG_OH は固定）
    if forced_oh:
        y_lo, y_hi = -0.5, 1.5
        x_range = (-0.5, 1.5)
        # x_vals を固定範囲に合わせて再計算
        x_vals = np.linspace(x_range[0], x_range[1], 200)
        # 理論線も更新
        if x_col == 'dG_OH':
            y1 = -x_vals + 1.72
            y2 = x_vals
        else:
            y1 = -0.5 * x_vals + 1.72
            y2 = 0.5 * x_vals
    else:
        y_candidates = [valid_data['limiting_potential'].min(), valid_data['limiting_potential'].max(), 1.23, y1.min(), y1.max(), y2.min(), y2.max()]
        y_lo, y_hi = _nice_bounds(min(y_candidates), max(y_candidates), step=0.5, pad_frac=0.08)
        x_lo, x_hi = x_range_pre
        x_range = (x_lo, x_hi)

    # 軸・アスペクト比
    ax.set_xlim(x_range)
    ax.set_ylim(y_lo, y_hi)
    if x_col == 'dG_OH':
        ax.set_xlabel(r'$\boldsymbol{\Delta}\mathbf{G}_{\mathbf{OH}^{*}}$ (eV)',
                      fontsize=20, fontweight='bold')
    else:
        ax.set_xlabel(f'{x_col} (eV)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Limiting Potential (V)', fontsize=20, fontweight='bold')
    if SHOW_TITLES:
        ax.set_title(f'ORR Volcano Plot: {x_col} vs Limiting Potential\n({legend_title}, iter0 - iter{max_iter})',
                     fontsize=20, fontweight='bold', pad=14)
    # 等尺は解除し、マージンを最小化
    ax.set_aspect('auto')
    ax.margins(x=0.02, y=0.02)

    # 目盛・凡例
    # 0.5間隔のメジャー目盛りを設定（グリッドはこの目盛りに揃える）
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 0.5間隔の目盛りに合わせてグリッドを表示
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='both', linestyle='--', color='gray', alpha=0.6)

    if color_type == "iter":
        ax.legend(fontsize=14, loc='upper right', framealpha=0.9)
    else:
        handles, labels = [], []
        for line in ax.lines:
            if line.get_label() and not line.get_label().startswith('_'):
                handles.append(line); labels.append(line.get_label())
        if handles:
            ax.legend(handles, labels, loc='upper right', framealpha=0.9, fontsize=14)

    # 保存
    fig.tight_layout()
    fig.savefig(output_file, dpi=500, bbox_inches='tight', pad_inches=0.05)
    print(f"Volcano plot saved: {output_file}")
    plt.close(fig)


def create_orr_trend_plot(data, x_col, y_col, colors, output_file, max_iter, color_type="iter", linear_regression=True):
    """ORR Trend Plotを作成"""
    print(f"\n=== Creating ORR Trend Plot: {x_col} vs {y_col} ===")
    
    # 有効なデータのみを使用
    valid_data = data.dropna(subset=[x_col, y_col])
    
    if len(valid_data) == 0:
        print(f"No valid data found for trend plot!")
        return
    
    plt.figure(figsize=(12, 12))
    
    if color_type == "iter":
        # イテレーションごとに色分け
        iter_order = get_iter_order(valid_data)
        color_map = setup_color_palette(max_iter)
        iter_colors = {iter_name: color_map.get(iter_name, 'gray') for iter_name in iter_order}
        
        for iter_name in iter_order:
            iter_data = valid_data[valid_data['iter'] == iter_name]
            if len(iter_data) > 0:
                plt.scatter(iter_data[x_col], iter_data[y_col],
                           c=iter_colors[iter_name], s=100, alpha=0.85,
                           edgecolors='black', linewidths=0.2,
                           label=f'{iter_name}')
        
        legend_title = 'Iterations'
        
    else:
        # 元素含有量でヒートマップ
        fraction_col = f'{color_type}_fraction'
        if fraction_col not in valid_data.columns:
            print(f"  Warning: {fraction_col} not found. Skipping trend plot colored by {color_type}.")
            return
        scatter = plt.scatter(valid_data[x_col], valid_data[y_col],
                            c=valid_data[fraction_col], cmap='RdYlBu_r',
                            s=100, alpha=0.85, edgecolors='black', linewidths=0.2)
        
        # カラーバーを追加
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label(f'{color_type.capitalize()} Fraction', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=20)
        legend_title = f'{color_type.capitalize()} Fraction'
    
    # 線形回帰
    if linear_regression and len(valid_data) > 1:
        # 3σ法で外れ値除去（線形回帰用）
        def remove_outliers_3sigma(data, x_col, y_col):
            """3σ法で外れ値を除去"""
            x_vals = data[x_col].values
            y_vals = data[y_col].values
            
            # x, y それぞれで3σ外れ値を特定
            x_mean, x_std = np.mean(x_vals), np.std(x_vals)
            y_mean, y_std = np.mean(y_vals), np.std(y_vals)
            
            # 3σ範囲内のデータを抽出
            x_mask = np.abs(x_vals - x_mean) <= 3 * x_std
            y_mask = np.abs(y_vals - y_mean) <= 3 * y_std
            combined_mask = x_mask & y_mask
            
            filtered_data = data[combined_mask]
            removed_count = len(data) - len(filtered_data)
            
            print(f"  3σ法で外れ値除去: {removed_count}個のデータを除去 ({len(data)} → {len(filtered_data)})")
            
            return filtered_data
        
        # 外れ値除去したデータで線形回帰
        regression_data = remove_outliers_3sigma(valid_data, x_col, y_col)
        
        if len(regression_data) > 1:
            # numpy を使った線形回帰（外れ値除去後のデータ）
            X_reg = regression_data[x_col].values
            y_reg = regression_data[y_col].values
            
            # 最小二乗法で回帰係数を計算
            A = np.vstack([X_reg, np.ones(len(X_reg))]).T
            slope, intercept = np.linalg.lstsq(A, y_reg, rcond=None)[0]
            
            # R²を計算（外れ値除去後のデータで）
            y_pred_reg = slope * X_reg + intercept
            ss_res = np.sum((y_reg - y_pred_reg) ** 2)
            ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 回帰線をプロット（全データ範囲で）
            x_range = np.linspace(valid_data[x_col].min() - 0.5, valid_data[x_col].max() + 0.5, 100)
            y_range = slope * x_range + intercept
            plt.plot(x_range, y_range, 'r-', linewidth=2, alpha=0.8, label='Linear fit (3σ filtered)')
            
            # 回帰式とR²を表示
            equation_text = f'{y_col} = {slope:.3f} × {x_col} + {intercept:.3f}\nR² = {r2:.3f} (3σ filtered, n={len(regression_data)})'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
        else:
            print("  警告: 3σ法による外れ値除去後のデータが不十分です")
    
    # 軸の設定
    x_margin = (valid_data[x_col].max() - valid_data[x_col].min()) * 0.1
    y_margin = (valid_data[y_col].max() - valid_data[y_col].min()) * 0.1
    plt.xlim(valid_data[x_col].min() - x_margin, valid_data[x_col].max() + x_margin)
    plt.ylim(valid_data[y_col].min() - y_margin, valid_data[y_col].max() + y_margin)
    
    plt.xlabel(f'{x_col} (eV)', fontsize=20, fontweight='bold')
    plt.ylabel(f'{y_col} (eV)', fontsize=20, fontweight='bold')
    if SHOW_TITLES:
        plt.title(f'ORR Trend Plot: {x_col} vs {y_col}\n({legend_title}, iter0 - iter{max_iter})', 
                  fontsize=20, fontweight='bold', pad=20)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 凡例
    if color_type == "iter":
        plt.legend(fontsize=16, loc='best', framealpha=0.9)
    else:
        # 回帰線の凡例のみ
        if linear_regression:
            regression_handles = [plt.Line2D([0], [0], color='red', linewidth=2)]
            regression_labels = ['Linear fit']
            plt.legend(regression_handles, regression_labels, loc='best', framealpha=0.9, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Trend plot saved: {output_file}")
    plt.close()


def create_all_orr_plots(combined_df, subdirs, max_iter):
    """全てのORRプロットを作成"""
    print(f"\n=== Creating All ORR Plots ===")
    
    # dG値を計算
    combined_df = calculate_dG_values(combined_df)
    
    # 有効なデータを確認
    valid_data = combined_df.dropna(subset=['dG_OH', 'dG_O', 'dG_OOH', 'limiting_potential'])
    
    if len(valid_data) == 0:
        print("No valid data found for ORR plots!")
        return
    
    # カラーパレット設定
    colors = setup_color_palette(max_iter)
    
    # =============================================================================
    # Volcano Plots
    # =============================================================================
    
    fraction_elements = get_available_fraction_elements(valid_data)

    # Volcano plots with iteration coloring and each element fraction coloring
    create_orr_volcano_plot(
        valid_data,
        'dG_OH',
        colors,
        subdirs['volcano_plot'] / f"volcano_dG_OH_vs_limiting_potential_iter0-{max_iter}.png",
        max_iter,
        color_type="iter"
    )

    for element in fraction_elements:
        element_key = element.lower()
        output_file = subdirs['volcano_plot'] / (
            f"volcano_dG_OH_vs_limiting_potential_{element_key}_heatmap_iter0-{max_iter}.png"
        )
        create_orr_volcano_plot(
            valid_data,
            'dG_OH',
            colors,
            output_file,
            max_iter,
            color_type=element_key
        )

    # Trend plots (iterative and per-element coloring)
    trend_pairs = [('dG_OH', 'dG_OOH'), ('dG_OH', 'dG_O')]
    for x_col, y_col in trend_pairs:
        create_orr_trend_plot(
            valid_data,
            x_col,
            y_col,
            colors,
            subdirs['trend_plot'] / f"trend_{x_col}_vs_{y_col}_iter0-{max_iter}.png",
            max_iter,
            color_type="iter"
        )

        for element in fraction_elements:
            element_key = element.lower()
            output_file = subdirs['trend_plot'] / (
                f"trend_{x_col}_vs_{y_col}_{element_key}_heatmap_iter0-{max_iter}.png"
            )
            create_orr_trend_plot(
                valid_data,
                x_col,
                y_col,
                colors,
                output_file,
                max_iter,
                color_type=element_key
            )
    
    # 統計サマリーを出力
    print_orr_statistics(valid_data, max_iter)
    
    return valid_data


def print_orr_statistics(valid_data, max_iter):
    """ORR統計サマリーを出力"""
    print(f"\n=== ORR Statistics Summary (iter0 - iter{max_iter}) ===")
    available_elements = get_available_fraction_elements(valid_data)
    
    # 基本統計
    for col in ['dG_OH', 'dG_O', 'dG_OOH', 'limiting_potential']:
        print(f"\n{col}:")
        print(f"  Mean: {valid_data[col].mean():.3f}")
        print(f"  Std: {valid_data[col].std():.3f}")
        print(f"  Min: {valid_data[col].min():.3f}")
        print(f"  Max: {valid_data[col].max():.3f}")
    
    # 最高性能サンプル
    best_samples = valid_data.nlargest(5, 'limiting_potential')
    print(f"\n=== Top 5 Limiting Potential Samples ===")
    for idx, row in best_samples.iterrows():
        fraction_parts = []
        for element in available_elements:
            col = fraction_column(element)
            if col in row and not pd.isna(row[col]):
                fraction_parts.append(f"{element}: {row[col]:.3f}")
        fraction_summary = ", ".join(fraction_parts) if fraction_parts else "Fractions: N/A"
        print(
            f"  LP: {row['limiting_potential']:.3f} V, "
            f"dG_OH: {row['dG_OH']:.3f}, dG_O: {row['dG_O']:.3f}, dG_OOH: {row['dG_OOH']:.3f}, "
            f"{fraction_summary}"
        )


# =============================================================================
# 元素含有量ヒートマップとPhase Diagram
# =============================================================================

def create_element_fraction_heatmaps(combined_df, subdirs, max_iter):
    """元素分率に基づくヒートマップ群を作成"""
    print(f"\n=== Creating Element Fraction Heatmaps ===")

    elements = get_available_fraction_elements(combined_df)
    if not elements:
        print("  No element fraction columns found. Skipping heatmap generation.")
        return {}

    heatmap_results = {}

    for element in elements:
        fraction_col = fraction_column(element)
        if fraction_col not in combined_df.columns:
            continue

        valid_data = combined_df.dropna(subset=['E_alloy_formation', 'overpotential', fraction_col]).copy()
        print(f"  {element}: {len(valid_data)} samples available for heatmap")

        if len(valid_data) == 0:
            continue

        alloy_formation_range = (
            valid_data['E_alloy_formation'].min(),
            valid_data['E_alloy_formation'].max()
        )
        overpot_range = (
            valid_data['overpotential'].min(),
            valid_data['overpotential'].max()
        )
        fraction_range = (
            valid_data[fraction_col].min(),
            valid_data[fraction_col].max()
        )

        print("    Data ranges:")
        print(f"      Alloy Formation Energy: {alloy_formation_range[0]:.3f} ~ {alloy_formation_range[1]:.3f} eV/atom")
        print(f"      Overpotential: {overpot_range[0]:.3f} ~ {overpot_range[1]:.3f} V")
        print(f"      {element} fraction: {fraction_range[0]:.3f} ~ {fraction_range[1]:.3f}")

        base_rgb = resolve_element_rgb(element)
        element_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{element.lower()}_fraction_cmap",
            [(1.0, 1.0, 1.0), base_rgb]
        )

        fig, ax = plt.subplots(figsize=(12, 12))
        scatter = ax.scatter(
            valid_data['E_alloy_formation'],
            valid_data['overpotential'],
            c=valid_data[fraction_col],
            cmap=element_cmap,
            s=100,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.2,
        )

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(f'{element} Fraction', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=20)

        ax.set_xlabel('Alloy Formation Energy (eV/atom)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Overpotential (V)', fontsize=20, fontweight='bold')
        if SHOW_TITLES:
            ax.set_title(
                f'Overpotential vs Alloy Formation Energy\nColored by {element} Fraction (iter0 - iter{max_iter})',
                fontsize=20,
                fontweight='bold',
                pad=20,
            )

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.tight_layout()

        heatmap_output = subdirs['heatmap'] / (
            f"overpotential_vs_alloy_formation_{element.lower()}_fraction_heatmap_iter0-{max_iter}.png"
        )
        fig.savefig(heatmap_output, dpi=500, bbox_inches='tight')
        print(f"    {element} fraction heatmap saved: {heatmap_output}")
        plt.close(fig)

        plt.figure(figsize=(12, 8))
        plt.hist(
            valid_data[fraction_col],
            bins=30,
            alpha=0.75,
            color=mcolors.to_hex(base_rgb),
            edgecolor='black',
            linewidth=0.5,
        )
        plt.xlabel(f'{element} Fraction', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=14, fontweight='bold')
        if SHOW_TITLES:
            plt.title(
                f'Distribution of {element} Fraction (iter0 - iter{max_iter})',
                fontsize=16,
                fontweight='bold',
                pad=20,
            )
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)

        mean_fraction = valid_data[fraction_col].mean()
        std_fraction = valid_data[fraction_col].std()
        if not np.isnan(mean_fraction):
            plt.axvline(mean_fraction, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_fraction:.3f}')
        if not np.isnan(std_fraction):
            plt.axvline(mean_fraction + std_fraction, color='orange', linestyle=':', linewidth=2,
                        label=f'Mean + Std: {mean_fraction + std_fraction:.3f}')
            plt.axvline(mean_fraction - std_fraction, color='orange', linestyle=':', linewidth=2,
                        label=f'Mean - Std: {mean_fraction - std_fraction:.3f}')

        plt.legend(fontsize=12)
        plt.tight_layout()

        dist_output = subdirs['heatmap'] / (
            f"{element.lower()}_fraction_distribution_iter0-{max_iter}.png"
        )
        plt.savefig(dist_output, dpi=300, bbox_inches='tight')
        print(f"    {element} fraction distribution saved: {dist_output}")
        plt.close()

        bin_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        valid_data[f'{element.lower()}_fraction_bin'] = pd.cut(
            valid_data[fraction_col], bins=5, labels=bin_labels
        )
        performance_summary = valid_data.groupby(f'{element.lower()}_fraction_bin').agg({
            'overpotential': ['count', 'mean', 'std', 'min'],
            'E_alloy_formation': ['mean', 'std'],
            fraction_col: ['mean', 'min', 'max']
        }).round(4)

        print(f"\n=== {element}含有量による性能分析 ===")
        print(performance_summary)

        best_performance = valid_data.nsmallest(10, 'overpotential')
        print(f"\n=== Top 10 Low Overpotential Samples ({element}) ===")
        print(f"Overpotential, Alloy Formation Energy, {element} Fraction:")
        for _, row in best_performance.iterrows():
            fraction_value = row.get(fraction_col, np.nan)
            fraction_str = f"{fraction_value:.3f}" if not np.isnan(fraction_value) else 'N/A'
            print(
                f"  {row['overpotential']:.3f} V, "
                f"{row['E_alloy_formation']:.3f} eV/atom, "
                f"{element}: {fraction_str}"
            )

        heatmap_results[element] = valid_data

    return heatmap_results

def create_phase_diagram(combined_df, subdirs, max_iter):
    """Pt-M 二元系を対象とした縦軸:形成エネルギー / 横軸:M分率の散布図を作成"""
    print(f"\n=== Creating Phase Diagram ===")

    available_metals = [
        (element, fraction_column(element))
        for element in get_available_fraction_elements(combined_df, include_pt=False)
        if fraction_column(element) in combined_df.columns
    ]

    required_columns = ['E_alloy_formation', 'pt_fraction', 'iter'] + [col for _, col in available_metals]
    valid_data = combined_df.dropna(subset=['E_alloy_formation', 'pt_fraction', 'iter']).copy()
    for _, col in available_metals:
        valid_data[col] = valid_data[col].fillna(0.0).astype(float)
    valid_data['pt_fraction'] = valid_data['pt_fraction'].fillna(0.0).astype(float)

    if valid_data.empty or not available_metals:
        print("No valid data found for phase diagram creation!")
        return None

    # Pt-M（二元）に絞る: Pt以外の分率が1種類のみ正のサンプルを抽出
    non_pt_counts = sum((valid_data[col] > 1e-6).astype(int) for _, col in available_metals)
    valid_data = valid_data[non_pt_counts == 1].copy()

    if valid_data.empty:
        print("No Pt-M binary samples available for phase diagram plots.")
        return None

    valid_data['metal'] = 'Unknown'
    valid_data['metal_fraction'] = 0.0
    for metal, col in available_metals:
        mask = valid_data[col] > 1e-6
        valid_data.loc[mask, 'metal'] = metal
        valid_data.loc[mask, 'metal_fraction'] = valid_data.loc[mask, col]

    metals_in_data = [metal for metal, _ in available_metals if (valid_data['metal'] == metal).any()]
    if not metals_in_data:
        print("Filtered dataset does not contain Pt-M binaries with positive fractions.")
        return None

    alloy_formation_range = (
        valid_data['E_alloy_formation'].min(),
        valid_data['E_alloy_formation'].max()
    )
    print(f"Valid samples for phase diagram: {len(valid_data)}")
    print("Data ranges by metal:")
    for metal in metals_in_data:
        metal_df = valid_data[valid_data['metal'] == metal]
        frac_min = metal_df['metal_fraction'].min()
        frac_max = metal_df['metal_fraction'].max()
        print(f"  {metal} fraction: {frac_min:.3f} ~ {frac_max:.3f}")
    print(f"  Alloy Formation Energy: {alloy_formation_range[0]:.3f} ~ {alloy_formation_range[1]:.3f} eV/atom")

    # ---- 図1: イテレーションカラー ----
    color_map = setup_color_palette(max_iter)
    unique_iters = sorted(valid_data['iter'].unique(), key=lambda x: int(str(x).replace('iter', '')))

    fig_width = 6 * len(metals_in_data)
    fig_iter, axes_iter = plt.subplots(1, len(metals_in_data), sharey=True, figsize=(fig_width, 6))
    if len(metals_in_data) == 1:
        axes_iter = [axes_iter]

    for ax, metal in zip(axes_iter, metals_in_data):
        metal_df = valid_data[valid_data['metal'] == metal]
        if metal_df.empty:
            ax.set_visible(False)
            continue

        iter_colors = metal_df['iter'].map(color_map).fillna('gray')
        ax.scatter(
            metal_df['metal_fraction'],
            metal_df['E_alloy_formation'],
            c=list(iter_colors),
            s=45,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.3,
        )

        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(f'{metal} Fraction', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if SHOW_TITLES:
            ax.set_title(f'Pt-{metal}', fontsize=18, fontweight='bold', pad=12)

    axes_iter[0].set_ylabel('Formation Energy (eV/atom)', fontsize=16, fontweight='bold')

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(iter_name, 'gray'),
                   markeredgecolor='black', markersize=9, label=iter_name)
        for iter_name in unique_iters
    ]
    if legend_handles:
        fig_iter.legend(legend_handles, [h.get_label() for h in legend_handles],
                        loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(legend_handles)),
                        fontsize=12, frameon=False)

    fig_iter.tight_layout(rect=(0, 0, 1, 0.94))
    iter_output = subdirs['phase_diagram'] / f"phase_diagram_ptm_iter_colors_iter0-{max_iter}.png"
    fig_iter.savefig(iter_output, dpi=500, bbox_inches='tight')
    print(f"Pt-M iteration-colored phase diagram saved: {iter_output}")
    plt.close(fig_iter)

    # ---- 図2: オーバーポテンシャルカラー ----
    if 'overpotential' not in valid_data.columns:
        print("No overpotential column found; skipping overpotential-colored phase diagram.")
        return valid_data

    over_data = valid_data.dropna(subset=['overpotential']).copy()
    if over_data.empty:
        print("No samples with overpotential values for colored phase diagram.")
        return valid_data

    over_min = over_data['overpotential'].min()
    over_max = over_data['overpotential'].max()

    fig_over, axes_over = plt.subplots(1, len(metals_in_data), sharey=True, figsize=(fig_width, 6))
    if len(metals_in_data) == 1:
        axes_over = [axes_over]

    scatters = []
    for ax, metal in zip(axes_over, metals_in_data):
        metal_df = over_data[over_data['metal'] == metal]
        if metal_df.empty:
            ax.set_visible(False)
            continue

        scatter = ax.scatter(
            metal_df['metal_fraction'],
            metal_df['E_alloy_formation'],
            c=metal_df['overpotential'],
            cmap='RdYlBu_r',
            vmin=over_min,
            vmax=over_max,
            s=45,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.3,
        )
        scatters.append(scatter)

        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(f'{metal} Fraction', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if SHOW_TITLES:
            ax.set_title(f'Pt-{metal}', fontsize=18, fontweight='bold', pad=12)

    axes_over[0].set_ylabel('Formation Energy (eV/atom)', fontsize=16, fontweight='bold')

    # tight_layout を後から呼ぶと色バーの領域が確保されず軸と重なってしまうため、
    # こちらで明示的に余白を調整してから色バーを作成する。
    fig_over.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.12, wspace=0.25)

    if scatters:
        cbar = fig_over.colorbar(
            scatters[-1], ax=axes_over, orientation='vertical', fraction=0.05, pad=0.04
        )
        cbar.set_label('Overpotential (V)', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
    over_output = subdirs['phase_diagram'] / f"phase_diagram_ptm_overpotential_iter0-{max_iter}.png"
    fig_over.savefig(over_output, dpi=500, bbox_inches='tight')
    print(f"Pt-M overpotential-colored phase diagram saved: {over_output}")
    plt.close(fig_over)

    return valid_data

# =============================================================================
# メイン実行関数
# =============================================================================

def main():
    """メイン実行関数"""
    args = parse_args()
    global SHOW_TITLES
    SHOW_TITLES = bool(args.title)
    
    print(f"Starting ORR catalyst data analysis (iter0 - iter{args.iter})...")
    print(f"Data directory: {args.base_path}")
    print(f"Output directory: {args.output_path}")
    
    # メイン分析実行
    combined_df = analyze_orr_catalyst_data(
        max_iter=args.iter,
        base_path=args.base_path,
        output_path=args.output_path,
        palette=args.palette
    )
    
    # 結果サマリー表示
    _display_analysis_summary(combined_df, args)
    
    # メモリクリーンアップ
    _cleanup_resources()
    
    print("Analysis script completed successfully!")


def _display_analysis_summary(combined_df, args):
    """分析結果のサマリーを表示"""
    if combined_df is not None and not combined_df.empty:
        output_path = Path(args.output_path)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Analyzed iterations: iter0 - iter{args.iter}")
        print(f"All figures saved to: {output_path}")
        print(f"Generated files organized by type:")
        
        # サブディレクトリごとにファイルを表示
        subdirs = {
            'histogram': 'Histogram Plots',
            'scatter_plot': 'Scatter Plots', 
            'violin_plot': 'Violin Plots',
            'box_plot': 'Box Plots',
            'heatmap': 'Heatmap Plots',
            'phase_diagram': 'Phase Diagrams',
            'structure_visualization': 'Structure Visualizations',
            'volcano_plot': 'ORR Volcano Plots',
            'trend_plot': 'ORR Trend Plots',
            'statistics': 'Statistics Files'
        }
        
        for subdir_name, display_name in subdirs.items():
            subdir_path = output_path / subdir_name
            if subdir_path.exists():
                files = list(subdir_path.glob(f"*iter0-{args.iter}*"))
                if files:
                    print(f"\n  {display_name}:")
                    for file in sorted(files):
                        if file.is_file():
                            size_mb = file.stat().st_size / (1024 * 1024)
                            print(f"    - {file.name} ({size_mb:.2f} MB)")
    else:
        print("No data available for analysis.")


def _cleanup_resources():
    """メモリとリソースのクリーンアップ"""
    import gc
    gc.collect()
    
    # matplotlibのクリーンアップ
    plt.close('all')
    matplotlib.pyplot.close('all')


if __name__ == "__main__":
    main()
