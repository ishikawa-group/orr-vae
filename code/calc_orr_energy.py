#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate energies of the ORR reaction
"""

import os, sys, json, time, numpy as np
from typing import Dict, Any, Tuple
from ase import Atom, Atoms
from ase.build import fcc111, bulk, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.vasp import Vasp
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter, ExpCellFilter
from ase.io import write

# Add custom module path
from tool import parallel_displacement, fix_lower_surface, set_initial_magmoms, auto_lmaxmix, my_calculator, convert_numpy_types

# ----------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------      
BASE_DIR      = f"result/matter_sim"

# Adsorption sites
SITES         = ["ontop", "bridge", "fcc"] 

# Molecules  shape
MOLECULES = {
    "OH" : Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, 0, 1.46), (0.939, 0, 1.705)]),  
    "H2" : Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
    "O2" : Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H"  : Atoms("H",   positions=[(0, 0, 0)]),
    "O"  : Atoms("O",   positions=[(0, 0, 0)])
}



# 構造パラメータ
SLAB_VACUUM   = 30.0      # Å
GAS_BOX       = 15.0      # Å
ADS_HEIGHT    = 2.0       # Å

# ----------------------------------------------------------------------
# calculation settings
# ----------------------------------------------------------------------
def optimize_gas(mol_name: str, gas_box: float, work_dir: str, calc_type: str = "mattersim") -> Tuple[Atoms, float]:
    mol = MOLECULES[mol_name].copy()
    mol.set_cell([gas_box, gas_box, gas_box])
    mol.set_pbc(True)
    mol.center()
    mol = set_initial_magmoms(mol, kind="gas", formula=mol_name)
    opt_mol = my_calculator(mol, "gas", calc_type=calc_type, calc_directory=work_dir)
    E_opt_gas = opt_mol.get_potential_energy()
    return opt_mol, E_opt_gas

def optimize_bulk(bulk: Atoms, work_dir: str, calc_type: str = "mattersim") -> Tuple[Atoms, float]:
    bulk_atoms = bulk.copy()
    bulk_atoms.set_pbc(True)
    bulk_atoms = set_initial_magmoms(bulk_atoms, kind="bulk")
    opt_bulk = my_calculator(bulk_atoms, "bulk", calc_type=calc_type, calc_directory=work_dir)
    auto_lmaxmix(opt_bulk)
    E_opt_bulk = opt_bulk.get_potential_energy()
    return opt_bulk.atoms, E_opt_bulk

def optimize_slab(opt_bulk: Atoms, work_dir: str, calc_type: str = "mattersim") -> Tuple[Atoms, float]:
    slab = opt_bulk.copy()
    slab.set_pbc(True)
    slab = parallel_displacement(slab, vacuum=SLAB_VACUUM)
    slab = fix_lower_surface(slab)
    slab = set_initial_magmoms(slab, kind="slab")
    opt_slab = my_calculator(slab, "slab", calc_type=calc_type, calc_directory=work_dir)
    auto_lmaxmix(opt_slab)
    E_opt_slab = opt_slab.get_potential_energy()
    return opt_slab, E_opt_slab

def calc_adsorption_on_site(opt_slab:Atoms, opt_mol: Atoms, site: str, work_dir: str, calc_type: str = "mattersim") -> Tuple[float, float]:
    """指定したサイトでの吸着構造を最適化し、エネルギーと時間を返す"""
    print(f"   Site: {site}")
    site_dir = os.path.join(work_dir, f"site_{site}") # ディレクトリ名を変更
    os.makedirs(site_dir, exist_ok=True)

    # 開始時間の記録
    t0 = time.time()

    # スラブをコピー (最適化済み)
    atoms = opt_slab.copy()
    # スラブの磁気モーメントは最適化済みなので再設定不要な場合が多いが、念のため
    atoms = set_initial_magmoms(atoms, kind="slab")

    # 分子をコピー (最適化済み)
    ads = opt_mol.copy()
    # ガスの磁気モーメントも同様
    ads = set_initial_magmoms(ads, kind="gas", formula=ads.get_chemical_formula()) # formulaを動的に取得
    ads.center()
    ads.set_pbc(False)
    ads.set_cell(None)

    # 吸着分子を追加
    slab_ads = atoms.copy()
    slab_ads.set_pbc(True)
    slab_ads = fix_lower_surface(slab_ads)
    add_adsorbate(slab_ads, ads, height=ADS_HEIGHT, position=site)

    # 計算機を設定 (my_calculatorは最適化を実行しない前提)
    slab_ads_calc = my_calculator(slab_ads, "slab", calc_type=calc_type, calc_directory=site_dir) # 出力先をサイトディレクトリに

    # エネルギー計算
    E_opt_slab_ads = slab_ads_calc.get_potential_energy()

    # 経過時間の記録
    dt = time.time() - t0

    # 結果の保存
    write(os.path.join(site_dir, f"opt_slab_ads_{site}.xyz"), slab_ads_calc)
    print(f"     E_total = {E_opt_slab_ads:.6f} eV, Time = {dt:.2f} s")

    return E_opt_slab_ads, dt

def calculate_all_molecules(opt_slab: Atoms, E_opt_slab: float, calc_type: str = "mattersim") -> Dict[str, Any]:
    """MOLECULESに含まれる全ての分子について吸着エネルギー計算を実行する"""

    # 結果を保存する辞書
    all_results = {}

    # BASE_DIRが存在することを確認
    os.makedirs(BASE_DIR, exist_ok=True)

    # 各分子について計算を実行
    for mol_name in MOLECULES.keys():
        print(f"=== Processing {mol_name} ===")

        # 分子ごとのディレクトリを作成
        mol_dir = os.path.join(BASE_DIR, mol_name)
        os.makedirs(mol_dir, exist_ok=True)
        gas_dir = os.path.join(mol_dir, f"{mol_name}_gas") # ガス計算用ディレクトリ
        os.makedirs(gas_dir, exist_ok=True)
        ads_dir = os.path.join(mol_dir, "adsorption") # 吸着計算用ディレクトリ
        os.makedirs(ads_dir, exist_ok=True)

        try:
            # 1. ガス分子の構造最適化 (毎回実行)
            print(f"1) Optimizing {mol_name} gas molecule...")
            opt_mol, E_opt_gas = optimize_gas(mol_name, GAS_BOX, gas_dir, calc_type)

            # 2. 各サイトでの吸着エネルギー計算
            print(f"2) Calculating adsorption energies at different sites...")
            sites_data = {}
            for site in SITES:
                E_total, elapsed_time = calc_adsorption_on_site(opt_slab, opt_mol, site, ads_dir, calc_type)
                sites_data[site] = {
                    "E_total": E_total,
                    "elapsed time(s)": elapsed_time
                }

            # 3. 各サイトの吸着エネルギーを計算し、最も安定なサイトを見つける
            best_site = None
            best_adsorption_energy = float('inf') # 無限大で初期化

            for site, data in sites_data.items():
                # 吸着エネルギーを計算
                E_ads = data["E_total"] - (E_opt_slab + E_opt_gas)
                sites_data[site]["E_ads"] = E_ads # 計算結果を辞書に追加

                # 最も安定なサイト（吸着エネルギーが最小）を更新
                if E_ads < best_adsorption_energy:
                    best_adsorption_energy = E_ads
                    best_site = site

            # 4. 結果を保存
            all_results[mol_name] = {
                "E_gas": E_opt_gas,
                "E_slab": E_opt_slab, # スラブエネルギーは引数から受け取る
                "best_site": best_site,
                "best_adsorption_energy": best_adsorption_energy, # 最安定の吸着エネルギー
                "sites_data": sites_data # 各サイトの詳細データ（E_adsを含む）
            }

            print(f"{mol_name}: 計算完了 - 最適サイト: {best_site}, 最安定吸着エネルギー: {best_adsorption_energy:.6f} eV")

        except Exception as e:
            print(f"{mol_name}の計算中にエラーが発生しました: {str(e)}")
            all_results[mol_name] = {"error": str(e)}

    # 全体の結果をJSONとして保存
    summary_file = os.path.join(BASE_DIR, "all_results.json")
    with open(summary_file, 'w') as f:
        all_results = convert_numpy_types(all_results) # NumPy型を変換
        json.dump(all_results, f, indent=2)

    print(f"\n全ての計算が完了しました。結果は {summary_file} に保存されています。")

    return all_results

# メイン処理として実行
if __name__ == "__main__":
    print("ORR関連分子の吸着計算を開始します")

    # 計算タイプを設定（デフォルト値または引数から取得できるようにしても良い）
    calc_type = "mattersim"

    # 0. ディレクトリ準備
    os.makedirs(BASE_DIR, exist_ok=True)
    bulk_dir = os.path.join(BASE_DIR, "Pt_bulk")
    slab_dir = os.path.join(BASE_DIR, "Pt_slab")
    os.makedirs(bulk_dir, exist_ok=True)
    os.makedirs(slab_dir, exist_ok=True)

    # 1. バルクの構造最適化 (最初に一度だけ)
    print("1) Optimizing Pt bulk...")
    Pt111 = fcc111("Pt", size=(4,4,4), a=4.0, vacuum=None, periodic=True)
    opt_bulk, E_opt_bulk = optimize_bulk(Pt111, bulk_dir, calc_type)

    # 2. スラブの構造最適化 (最初に一度だけ)
    print("2) Optimizing Pt(111) slab...")
    opt_slab, E_opt_slab = optimize_slab(opt_bulk, slab_dir, calc_type)

    # 3. 全分子の吸着計算を実行
    results = calculate_all_molecules(opt_slab, E_opt_slab, calc_type)

    # 4. 結果の簡単なサマリーを表示
    print("\n=== 計算結果サマリー ===")
    for mol_name, data in results.items():
        if "error" in data:
            print(f"{mol_name}: エラー発生")
        else:
            print(f"{mol_name}: 最適サイト = {data['best_site']}, 最安定吸着エネルギー = {data['best_adsorption_energy']:.6f} eV")