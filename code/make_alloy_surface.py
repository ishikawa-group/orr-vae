#!/usr/bin/env python
import os
import argparse
from ase.build import fcc100
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
import numpy as np

# --- コマンドライン引数の定義 ---
parser = argparse.ArgumentParser(description="fcc100表面の4元素系合金（Pt, Pd, Cu）を生成")
parser.add_argument("--num", type=int, default=1,
                    help="生成する構造の数（デフォルトは1）")
args = parser.parse_args()

# --- パラメータ設定 ---
size = [5, 5, 4]
vacuum = None
lattice_const = 4.0
alloy_elements = ["Pt", "Ag", "Au"]

# --- 出力先ディレクトリの設定 ---
# 相対パスで "data" フォルダを指定（存在しない場合は作成）
data_dir = os.path.join(".", "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 出力ファイルのパス（iter0_surf.jsonをdataフォルダ内に出力）
db_path = os.path.join(data_dir, "iter0_structure.json")
if os.path.exists(db_path):
    os.remove(db_path)
db = connect(db_path)

# --- 指定された数だけ構造生成 ---
for i in range(args.num):
    # fcc100表面の作成（基本はPtの構造）
    surf = fcc100(symbol="Pt", 
                  size=size, 
                  a=lattice_const,
                  vacuum=vacuum, 
                  periodic=True)

    # --- 合金組成の均等配置 ---
    natoms = len(surf)  # 表面内の原子数
    base_count = natoms // len(alloy_elements)  # 各元素の基本個数
    leftover = natoms % len(alloy_elements)      # 割り切れない余り分

    # 各元素の原子番号を均等に配置するリストを作成
    alloy_list = []
    for element in alloy_elements:
        alloy_list.extend([atomic_numbers[element]] * base_count)
    for j in range(leftover):
        alloy_list.append(atomic_numbers[alloy_elements[j]])

    # ランダムにシャッフルして各元素をランダム配置
    np.random.shuffle(alloy_list)
    surf.set_atomic_numbers(alloy_list)

    # --- EMT計算器の設定（FutureWarning解消のため calc 属性を直接代入） ---
    surf.calc = EMT()

    # --- データベースへの書き込み ---
    data = {"chemical_formula": surf.get_chemical_formula(), "run": i}
    db.write(surf, data=data)
