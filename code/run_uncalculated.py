#!/usr/bin/env python
import json
import os
import subprocess
import sys
from ase.db import connect

def get_surface_ids(surf_file):
    """表面構造ファイルからすべてのユニークIDを取得する"""
    unique_ids = []
    try:
        db = connect(surf_file)
        for row in db.select():
            if hasattr(row, 'unique_id'):
                unique_ids.append(row.unique_id)
            else:
                # 数値IDの場合
                unique_ids.append(str(row.id))
    except Exception as e:
        print(f"表面構造ファイルの読み込みエラー: {e}")
        sys.exit(1)
    
    return unique_ids

def get_calculated_ids(reaction_energy_file):
    """反応エネルギーファイルから既に計算済みのIDを取得する"""
    calculated_ids = []
    
    if not os.path.exists(reaction_energy_file):
        return calculated_ids
    
    try:
        with open(reaction_energy_file, 'r') as f:
            data = json.load(f)
            
        # 単一のエントリか配列かを確認
        if isinstance(data, list):
            calculated_ids = [entry.get("unique_id") for entry in data]
        elif isinstance(data, dict) and "unique_id" in data:
            calculated_ids = [data.get("unique_id")]
    except Exception as e:
        print(f"反応エネルギーファイルの読み込みエラー: {e}")
    
    return calculated_ids

def run_calculation(unique_id, surf_file, reaction_energy_file):
    """指定されたIDに対して計算を実行する"""
    print(f"ID {unique_id} の計算を開始します...")
    
    cmd = [
        "python", 
        "/Users/wakamiya/Documents/ORR_catalyst_generator/calc_orr_reaction_energy.py",
        "--id", unique_id,
        "--slab_file", surf_file,
        "--out_json", reaction_energy_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"ID {unique_id} の計算が完了しました。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ID {unique_id} の計算中にエラーが発生しました: {e}")
        return False

def main():
    # ファイルパスの設定
    surf_file = "data/iter0_surf.json"
    reaction_energy_file = "data/reaction_energy.json"
    
    # コマンドライン引数から別のファイルパスを指定可能にする
    if len(sys.argv) > 1:
        surf_file = sys.argv[1]
    if len(sys.argv) > 2:
        reaction_energy_file = sys.argv[2]
    
    print(f"表面構造ファイル: {surf_file}")
    print(f"反応エネルギーファイル: {reaction_energy_file}")
    
    # 表面構造IDの取得
    surface_ids = get_surface_ids(surf_file)
    print(f"表面構造ファイル内のID数: {len(surface_ids)}")
    
    # 計算済みIDの取得
    calculated_ids = get_calculated_ids(reaction_energy_file)
    print(f"既に計算済みのID数: {len(calculated_ids)}")
    
    # 未計算のIDを特定
    uncalculated_ids = [id for id in surface_ids if id not in calculated_ids]
    print(f"未計算のID数: {len(uncalculated_ids)}")
    
    if not uncalculated_ids:
        print("すべてのIDが既に計算済みです。")
        return
    
    # 未計算のIDに対して計算を実行
    success_count = 0
    for unique_id in uncalculated_ids:
        if run_calculation(unique_id, surf_file, reaction_energy_file):
            success_count += 1
    
    print(f"計算完了: {success_count}/{len(uncalculated_ids)} 件の計算が成功しました。")

if __name__ == "__main__":
    main()