#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ase.db import connect

# 自作モジュールからORR過電圧計算関数をインポート
from orr_overpotential_calculator import calc_orr_overpotential

def main():
    parser = argparse.ArgumentParser(description="ORR過電圧計算ツール")
    parser.add_argument('--bulk_db', required=True, help='バルク構造を含むASE DB (JSON)')
    parser.add_argument('--out_json', required=True, help='結果出力用JSONファイル')
    parser.add_argument('--unique_id', required=True, help='処理する構造のユニークID')
    parser.add_argument('--base_dir', default="./result", help='計算用ベースディレクトリ')
    parser.add_argument('--force', default=True, help='強制的に計算を実行')
    parser.add_argument('--log_level', default="INFO", help='ログレベル')
    parser.add_argument('--calc_type', default="mace", help='計算機タイプ')
    parser.add_argument('--yaml_path', help='VASP設定YAMLファイルパス')
    args = parser.parse_args()

    # 出力ディレクトリの作成
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    
    # 既存の結果を読み込む
    results = []
    ids = []
    if Path(args.out_json).exists() and Path(args.out_json).stat().st_size > 0:
        with open(args.out_json) as f:
            existing = json.load(f)
            results = existing if isinstance(existing, list) else [existing]
            ids = [entry.get('unique_id') for entry in results]
    
    uid = args.unique_id
    
    # Create and save initial entry with null values before starting calculations
    if uid not in ids:
        initial_entry = {
            'unique_id': uid,
            'overpotential': None
        }
        results.append(initial_entry)
        
        # Save initial state to JSON
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Added initial entry for {uid}")
        
        ids.append(uid)
    
    # DBからバルク構造を取得
    db = connect(args.bulk_db)
    
    # いくつかの方法でIDを検索
    row = None
    for method in [
        lambda: list(db.select(f'unique_id="{uid}"')),
        lambda: list(db.select(unique_id=uid)),
        lambda: [r for r in db.select() if 'unique_id' in r.key_value_pairs and r.key_value_pairs['unique_id'] == uid],
        lambda: [db.get(int(uid))]
    ]:
        try:
            rows = method()
            if rows:
                row = rows[0]
                break
        except (ValueError, KeyError):
            continue
    
    if row is None:
        print(f"Error: unique_id '{uid}' not found in database")
        sys.exit(1)
    
    # バルク構造を取得&表面情報の追加
    bulk_atoms = row.toatoms(add_additional_information=True)
    d = bulk_atoms.info.pop("data", {})
    bulk_atoms.info["adsorbate_info"] = d["adsorbate_info"]
    
    # 計算ディレクトリの設定
    calc_dir = Path(args.base_dir) / uid

    # 組成比の計算
    atomic_numbers = bulk_atoms.get_atomic_numbers()
    total_atoms = len(atomic_numbers)
    ni_count = sum(1 for num in atomic_numbers if num == 28)
    pt_count = sum(1 for num in atomic_numbers if num == 78)

    ni_fraction = ni_count / total_atoms
    pt_fraction = pt_count / total_atoms
    
    # ORR吸着サイト定義
    orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
        "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)], #ontop, bridge, fcc, hcp
        "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
        "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    }
    
    # ORR過電圧の計算（修正部分）
    result = calc_orr_overpotential(
        bulk=bulk_atoms,
        outdir=str(calc_dir),
        force=args.force,
        log_level=args.log_level,
        calc_type=args.calc_type,
        adsorbates=orr_adsorbates,
        yaml_path=args.yaml_path
    )
    
    # 必要な値を辞書から取得
    eta = result["eta"]
    limiting_potential = 1.23 - result["eta"] 
    diffG_U0 = result["diffG_U0"]
    diffG_eq = result["diffG_eq"]
    
    # 結果の更新 - 要求された最小限の情報のみを含む
    entry = {
        'unique_id': uid,
        'overpotential': eta,
        'limiting_potential': limiting_potential,
        'diffG_U0': diffG_U0,
        'diffG_eq': diffG_eq,
        'ni_fraction': float(ni_fraction),  # 追加
        'pt_fraction': float(pt_fraction),  # 追加
        'chemical_formula': bulk_atoms.get_chemical_formula()  # 追加
    }
    
    # 既存エントリの更新または新規追加
    updated = False
    for i, existing_entry in enumerate(results):
        if existing_entry.get('unique_id') == uid:
            results[i] = entry
            updated = True
            break
    
    if not updated:
        results.append(entry)
    
    # 結果を保存
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"ORR overpotential for {uid}: {eta:.3f} V")
    print(f"Reaction Free Energy Change at U=0V: {diffG_U0}")
    print(f"Reaction Free Energy Change at U=1.23V: {diffG_eq}")

if __name__ == '__main__':
    main()