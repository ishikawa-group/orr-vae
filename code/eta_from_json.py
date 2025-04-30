#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from ase.db import connect

# 自作モジュールからORR過電圧計算関数をインポート
sys.path.append(str(Path(__file__).parent))
from calc_orr_overpotential import calc_orr_overpotential

def main():
    parser = argparse.ArgumentParser(description="ORR過電圧計算ツール")
    parser.add_argument('--bulk_db', required=True, help='バルク構造を含むASE DB (JSON)')
    parser.add_argument('--out_json', required=True, help='結果出力用JSONファイル')
    parser.add_argument('--unique_id', required=True, help='処理する構造のユニークID')
    parser.add_argument('--base_dir', default="./result", help='計算用ベースディレクトリ')
    parser.add_argument('--force', default=True, help='強制的に計算を実行')
    parser.add_argument('--log_level', default="INFO", help='ログレベル')
    parser.add_argument('--calc_type', default="mattersim", help='計算機タイプ')
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
    
    # バルク構造を取得
    bulk_atoms = row.toatoms()
    
    # 計算ディレクトリの設定
    calc_dir = Path(args.base_dir) / uid
    
    # ORR過電圧の計算 - 複雑な計算ロジックがcalc_orr_overpotentialに隠蔽されている
    eta = calc_orr_overpotential(
        bulk=bulk_atoms,
        base_dir=str(calc_dir),
        force=args.force,
        log_level=args.log_level,
        calc_type=args.calc_type
    )
    
    # 結果の更新 - 要求された最小限の情報のみを含む
    entry = {
        'unique_id': uid,
        'overpotential': eta
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

if __name__ == '__main__':
    main()