#!/usr/bin/env python
import os
import argparse
import uuid
from pathlib import Path  # ← これを追加
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
import numpy as np
from tool import vegard_lattice_constant

# --- コマンドライン引数の定義 ---
parser = argparse.ArgumentParser(description="fcc100表面の合金を生成")
parser.add_argument("--num", type=int, default=128,
                        help="Number of structures to generate (default: 128)")
parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "data"),  
                        help="Output directory (default: ./data)")
args = parser.parse_args()

# --- パラメータ設定 ---
size = [4, 4, 4]
vacuum = None
alloy_elements = ["Pt", "Ni"]

# --- 出力先ディレクトリの設定 ---
# args.output_dirを使用
data_dir = args.output_dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 出力ファイルのパス（iter0_structures.jsonを指定されたフォルダ内に出力）
db_path = os.path.join(data_dir, "iter0_structures.json")
#if os.path.exists(db_path):
#    os.remove(db_path)
db = connect(db_path)

# ランダムシード設定（再現性のため）
np.random.seed(42)

print(f"Generating {args.num} random alloy structures...")

# --- 指定された数だけ構造生成 ---
for i in range(args.num):
    # Ni置換率をランダムに決定（1-100%）
    ni_fraction = np.random.uniform(1/64, 63/64)
    pt_fraction = 1.0 - ni_fraction
    fractions = [pt_fraction, ni_fraction]
    
    # Vegard法で格子定数を計算
    lattice_const = vegard_lattice_constant(alloy_elements, fractions)
    
    # fcc111Bulkの作成（基本はPtの構造）
    bulk = fcc111(symbol="Pt", 
                  size=size, 
                  a=lattice_const,
                  vacuum=vacuum, 
                  periodic=True)

    # --- 合金組成のランダム配置 ---
    natoms = len(bulk)  # 総原子数 (4*4*4=64)
    n_ni = int(round(natoms * ni_fraction))
    n_pt = natoms - n_ni

    # 原子番号リストを作成
    alloy_list = ([atomic_numbers["Pt"]] * n_pt + 
                 [atomic_numbers["Ni"]] * n_ni)

    # ランダムにシャッフルして各元素をランダム配置
    np.random.shuffle(alloy_list)
    bulk.set_atomic_numbers(alloy_list)

    # --- EMT計算器の設定（FutureWarning解消のため calc 属性を直接代入） ---
    bulk.calc = EMT()

    # --- 表面情報の取得 ---
    ads_info = bulk.info["adsorbate_info"]

    # --- データベースへの書き込み ---
    data = {
        "chemical_formula": bulk.get_chemical_formula(), 
        "ni_fraction": float(ni_fraction),
        "pt_fraction": float(pt_fraction),
        "lattice_constant": float(lattice_const),
        "run": i,
        "adsorbate_info": ads_info
    }
    
    db.write(bulk, data=data)
    
    if (i+1) % 10 == 0:
        print(f"Generated {i+1}/{args.num} structures")

print(f"Structures saved to {db_path}")