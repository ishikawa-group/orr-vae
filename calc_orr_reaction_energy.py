#!/usr/bin/env python
import argparse
import json
import os
import numpy as np  # スクリプト冒頭でNumPyをインポート

from tool import fix_lower_surface
from tool import get_overpotential_orr

from ase import Atoms
from ase.build import add_adsorbate
from ase.db import connect
from ase.calculators.morse import MorsePotential
from ase.optimize import LBFGS

# ------------------------
# 関数定義
# ------------------------

def optimize_atoms(atoms, fmax=0.1, steps=200):
    """
    与えられた Atoms オブジェクトに対して MorsePotential をセットし、
    LBFGS によるジオメトリ最適化を実施して最適化後のエネルギーを返す。
    
    Args:
        atoms: ASE Atoms オブジェクト
        fmax: 収束判定用最大力（eV/Å、デフォルト: 0.1）
        steps: 最大最適化ステップ数（デフォルト: 200）
        
    Returns:
        (energy, optimized_atoms)
    """
    atoms = atoms.copy()
    # FutureWarning対応: set_calculator()の代わりにcalc属性を使用
    atoms.calc = MorsePotential()
    optimizer = LBFGS(atoms)
    optimizer.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    return energy, atoms


def prepare_gas_phase(mol_name, vacuum=20.0):
    """
    ガス相分子を、指定された真空セル（立方体セル：vacuum * vacuum * vacuum）にセットし中心化する。
    
    Args:
        mol_name: 分子名（例："O2", "H2", "H2O", "HO2", "OH"）
        vacuum: セルサイズ（Å、デフォルト: 20.0）
        
    Returns:
        ガス相分子の Atoms オブジェクト
    """
    atoms = Atoms(mol_name)
    cell = [vacuum, vacuum, vacuum]
    atoms.set_cell(cell)
    atoms.center()
    atoms.set_pbc(False)
    return atoms


def prepare_adsorbed_system(slab, adsorbate_name, height=1.5, offset=(0.5, 0.5)):
    """
    スラブに対して、指定された吸着種を on-top（スラブ上部中央付近）に追加した構造を返す。
    
    Args:
        slab: スラブとなる ASE Atoms オブジェクト
        adsorbate_name: 吸着種の名前（例："O", "O2", "H2", "H2O", "HO2", "OH"）
        height: 吸着種とスラブ表面間の垂直距離（Å、デフォルト: 1.5）
        offset: スラブ上での吸着位置オフセット（x, y、デフォルト: (0.5, 0.5)）
        
    Returns:
        吸着状態の Atoms オブジェクト（スラブ+吸着種）
    """
    slab_ads = slab.copy()
    slab_ads = fix_lower_surface(slab_ads)
    slab_ads.set_pbc(True)
    adsorbate = Atoms(adsorbate_name)
    add_adsorbate(slab_ads, adsorbate, height=height, offset=offset)
    return slab_ads


def convert_numpy_types(obj):
    """NumPy型を標準Python型に変換する関数"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ------------------------
# メイン処理
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MorsePotential を用いた反応エネルギー計算スクリプト\n"
                    "反応式:\n"
                    " (1) ΔE1: O2(g) + * → O2*\n"
                    " (2) ΔE2: O2* + H+ + e- → OOH*\n"
                    " (3) ΔE3: OOH* + H+ + e- → O* + H2O(g)\n"
                    " (4) ΔE4: O* + H+ + e- → OH*\n"
                    " (5) ΔE5: OH* + H+ + e- → * + H2O(g)"
    )
    parser.add_argument("--id", required=True, help="一意のID（unique id）")
    # slab_file は ase.db のファイル（JSON形式）とする
    parser.add_argument("--slab_file", default="data/iter0_surf.json",
                        help="スラブ構造が格納されている ase.db 形式のファイル")
    parser.add_argument("--out_json", default="data/reaction_energy.json",
                        help="計算結果を保存する JSON ファイル名")
    args = parser.parse_args()

    unique_id = args.id
    slab_file = args.slab_file

    # ---------------
    # ガス相エネルギー計算
    # ---------------
    gas_molecules = ["O2", "H2", "H2O", "HO2", "OH"]
    gas_energies = {}
    print("\n=== ガス相分子の最適化計算を実施中... ===")
    for mol in gas_molecules:
        print(f"  分子 {mol} の最適化中...")
        E, _ = optimize_atoms(prepare_gas_phase(mol))
        gas_energies[mol] = E
        print(f"  {mol}: {E:.4f} eV")

    # ---------------
    # スラブ構造の読み込み（ase.db 形式の JSON ファイル）
    # ---------------
    print(f"\n=== スラブ構造データ ({slab_file}) の読み込み ===")
    if not os.path.isfile(slab_file):
        raise FileNotFoundError(f"{slab_file} が存在しません。")
    
    db = connect(slab_file)
    
    # データベース内の全エントリのIDを表示して確認
    # print("データベース内の利用可能なID:")
    available_ids = []
    for row in db.select():
        if hasattr(row, 'unique_id'):
            # print(f"- unique_id: {row.unique_id}")
            available_ids.append(str(row.unique_id))
        else:
            # print(f"- id: {row.id}")
            available_ids.append(str(row.id))
    
    if not available_ids:
        print("警告: データベースにエントリが見つかりません。")
    
    # 複数の方法でIDを試す
    slab_row = None
    
    try:
        # 文字列として検索
        slab_row = db.get(unique_id=str(unique_id))
    except:
        try:
            # 整数として検索
            slab_row = db.get(unique_id=int(unique_id))
        except:
            try:
                # id として検索
                slab_row = db.get(id=int(unique_id))
            except:
                # 全て失敗した場合
                print(f"エラー: ID {unique_id} に一致するスラブ構造が見つかりません。")
                print(f"利用可能なID: {', '.join(available_ids)}")
                print("正しいIDを指定するか、データベースファイルを確認してください。")
                exit(1)
    
    print(f"ID {unique_id} のスラブ構造を読み込みました。")
    slab = slab_row.toatoms()

    # ---------------
    # スラブ系計算タスク
    # ---------------
    tasks = [
        ("slab", slab),
        ("slab+O", prepare_adsorbed_system(slab, "O")),
        ("slab+O2", prepare_adsorbed_system(slab, "O2")),
        ("slab+H2", prepare_adsorbed_system(slab, "H2")),
        ("slab+H2O", prepare_adsorbed_system(slab, "H2O")),
        ("slab+OOH", prepare_adsorbed_system(slab, "HO2")),
        ("slab+OH", prepare_adsorbed_system(slab, "OH"))
    ]

    slab_energies = {}
    total_tasks = len(tasks)
    print("\n=== スラブ系の最適化計算を実施中... ===")
    for i, (label, system) in enumerate(tasks):
        print(f"  {i+1}/{total_tasks}: {label} の最適化中...")
        E, _ = optimize_atoms(system)
        slab_energies[label] = E
        print(f"  {label}: {E:.4f} eV")

    # ---------------
    # エネルギーの取り出しと反応エネルギー算出
    # ---------------
    # ガス相エネルギー
    E_O2  = gas_energies["O2"]
    E_H2  = gas_energies["H2"]
    E_H2O = gas_energies["H2O"]

    # スラブ系エネルギー
    E_slab     = slab_energies["slab"]
    E_slab_O   = slab_energies["slab+O"]
    E_slab_O2  = slab_energies["slab+O2"]
    E_slab_H2O = slab_energies["slab+H2O"]
    E_slab_OOH = slab_energies["slab+OOH"]
    E_slab_OH  = slab_energies["slab+OH"]

    # ---------------
    # 反応エネルギー計算
    # 反応式：
    # (1) ΔE1: O2(g) + * → O2*
    # (2) ΔE2: O2* + H+ + e- → OOH*
    # (3) ΔE3: OOH* + H+ + e- → O* + H2O(g)
    # (4) ΔE4: O* + H+ + e- → OH*
    # (5) ΔE5: OH* + H+ + e- → * + H2O(g)
    deltaE1 = E_slab_O2 - (E_slab + E_O2)
    deltaE2 = E_slab_OOH - (E_slab_O2 + 0.5 * E_H2)
    deltaE3 = (E_slab_O + E_H2O) - (E_slab_OOH + 0.5 * E_H2)
    deltaE4 = E_slab_OH - (E_slab_O + 0.5 * E_H2)
    deltaE5 = E_slab_H2O - (E_slab_OH + 0.5 * E_H2)

    deltaEs = [deltaE2, deltaE3, deltaE4, deltaE5]
    overpotential = get_overpotential_orr(deltaEs, verbose=True)

    print("\n=== 反応エネルギーの計算結果 ===")
    print(f"  ΔE1 (O2吸着)      : {deltaE1:.4f} eV")
    print(f"  ΔE2 (第一電子移動): {deltaE2:.4f} eV")
    print(f"  ΔE3 (O-O結合開裂) : {deltaE3:.4f} eV")
    print(f"  ΔE4 (第三電子移動): {deltaE4:.4f} eV")
    print(f"  ΔE5 (第四電子移動): {deltaE5:.4f} eV")
    print(f"Calculated ORR Overpotential: {overpotential:.3f} eV")


# JSON 形式で結果を保存
# ---------------
# unique_idを文字列に変換
if isinstance(unique_id, np.integer):
    unique_id = str(int(unique_id))
else:
    unique_id = str(unique_id)
    
results = {
    "unique_id": unique_id,
    "gas_energies": gas_energies,
    "slab_energies": slab_energies,
    "reaction_energies": {
        "deltaE1": deltaE1,
        "deltaE2": deltaE2,
        "deltaE3": deltaE3,
        "deltaE4": deltaE4,
        "deltaE5": deltaE5,
    "η": overpotential
    }
}

# NumPy型を変換
results = convert_numpy_types(results)

out_json = args.out_json

# 出力ディレクトリが存在しない場合は作成
out_dir = os.path.dirname(out_json)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 既存のJSONファイルがあるか確認
existing_data = []
if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
    try:
        with open(out_json, 'r') as f:
            existing_data = json.load(f)
            
        # 読み込んだデータが配列でない場合は配列に変換
        if not isinstance(existing_data, list):
            existing_data = [existing_data]
    except json.JSONDecodeError:
        print(f"警告: {out_json} の読み込みに失敗しました。新しいファイルを作成します。")
        existing_data = []

# 既存のデータから同じIDのエントリを探す
existing_ids = [entry.get("unique_id") for entry in existing_data]
if unique_id in existing_ids:
    # 同じIDのエントリがある場合は上書き
    index = existing_ids.index(unique_id)
    existing_data[index] = results
    print(f"ID {unique_id} の既存データを更新しました。")
else:
    # 新規エントリとして追加
    existing_data.append(results)

# 更新したデータを書き込み
with open(out_json, "w") as f:
    json.dump(existing_data, f, indent=4)

print(f"\n=== 計算結果を {out_json} に保存しました ===")