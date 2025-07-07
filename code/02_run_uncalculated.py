#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未計算の構造に対してORR過電圧計算を実行するスクリプト
"""
import os
import json
import subprocess
import sys
import shutil
import argparse
from pathlib import Path
from ase.db import connect

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='未計算構造の過電圧計算')
    parser.add_argument('--iter', type=int, default=0,
                       help='イテレーション番号 (default: 0)')
    parser.add_argument('--base_data_dir', type=str, 
                       default=str(Path(__file__).parent / "data"),
                       help='データディレクトリのパス')
    parser.add_argument('--calc_script', type=str,
                       default=str(Path(__file__).parent / "02_calculate_overpotentials.py"),
                       help='計算スクリプトのパス')
    parser.add_argument('--temp_base_dir', type=str,
                       default=str(Path(__file__).parent / "result" / "test"),
                       help='一時ディレクトリのベースパス')
    parser.add_argument('--calc_type', type=str, default='mace',
                       choices=['mace', 'vasp', 'emt'],
                       help='計算タイプ (default: mace)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='ログレベル (default: INFO)')
    parser.add_argument('--force', action='store_true',
                       help='強制実行フラグ')
    parser.add_argument('--keep_temp', action='store_true',
                       help='一時ディレクトリを保持')
    return parser.parse_args()

def main():
    # コマンドライン引数を取得
    args = parse_args()
    
    # パラメータの設定
    ITER = args.iter
    base_data_dir = args.base_data_dir
    calc_script = args.calc_script
    temp_base_dir = args.temp_base_dir
    
    # 入力ファイル指定（動的に生成）
    bulk_file = os.path.join(base_data_dir, f"iter{ITER}_structures.json")
    calculation_result_file = os.path.join(base_data_dir, f"iter{ITER}_calculation_result.json")

    print(f"=== 未計算構造の過電圧計算 (iter{ITER}) ===")
    print(f"構造ファイル: {bulk_file}")
    print(f"結果ファイル: {calculation_result_file}")
    print(f"計算スクリプト: {calc_script}")
    print(f"計算タイプ: {args.calc_type}")
    print(f"ログレベル: {args.log_level}")
    print(f"強制実行: {args.force}")

    # ファイル存在チェック
    if not os.path.exists(bulk_file):
        print(f"エラー: 構造ファイルが見つかりません: {bulk_file}")
        sys.exit(1)
    
    if not os.path.exists(calc_script):
        print(f"エラー: 計算スクリプトが見つかりません: {calc_script}")
        sys.exit(1)

    # calculation_result_file のディレクトリを作成
    os.makedirs(os.path.dirname(calculation_result_file) or ".", exist_ok=True)

    # calculation_result_file が存在しない or 空の場合は空リストを初期化
    if not os.path.exists(calculation_result_file) or os.path.getsize(calculation_result_file) == 0:
        with open(calculation_result_file, 'w') as f:
            json.dump([], f)

    # Bulk DB から unique_id リストを抽出
    try:
        db = connect(bulk_file)
    except Exception as e:
        print(f"エラー: bulk DB に接続できませんでした({e})")
        sys.exit(1)

    bulk_ids = []
    for row in db.select():
        if hasattr(row, 'unique_id') and row.unique_id is not None:
            bulk_ids.append(str(row.unique_id))
        else:
            # フィールド unique_id がなければ id を使用
            bulk_ids.append(str(row.id))

    print(f"総構造数: {len(bulk_ids)}")

    # calculation_result_file の読み込み
    with open(calculation_result_file, 'r') as f:
        reaction_data = json.load(f)
    reaction_ids = [str(entry.get('unique_id', entry.get('id'))) for entry in reaction_data]

    print(f"計算済み構造数: {len(reaction_ids)}")

    # 最初の未計算 unique_id を取得
    uncalculated = [uid for uid in bulk_ids if uid not in reaction_ids]
    if not uncalculated:
        print("未計算の構造はありません。")
        sys.exit(0)

    uid = uncalculated[0]
    print(f"未計算構造数: {len(uncalculated)}")
    print(f"次に計算する構造: unique_id={uid}")

    # 一時ディレクトリを動的に生成
    temp_dir = os.path.join(temp_base_dir, f"iter{ITER}_{uid}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"一時ディレクトリを作成しました: {temp_dir}")

    # ORR過電圧計算スクリプトを呼び出し
    cmd = [
        "python3",
        calc_script,
        "--bulk_db", bulk_file,
        "--unique_id", uid,
        "--out_json", calculation_result_file,
        "--outdir", temp_dir,
        "--log_level", args.log_level,
        "--calc_type", args.calc_type
    ]
    
    # forceオプションの追加
    if args.force:
        cmd.extend(["--force", "True"])

    print(f"実行コマンド: {' '.join(cmd)}")
    print(f"unique_id={uid} の計算を開始します...")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"エラー: unique_id={uid} の計算に失敗しました (exit code {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f"unique_id={uid} の計算が完了しました。")

    # 計算後、calculation_result_file を再読み込みして成功数を表示
    with open(calculation_result_file, 'r') as f:
        updated = json.load(f)
    updated_ids = [str(entry.get('unique_id', entry.get('id'))) for entry in updated]
    success = updated_ids.count(uid)

    print(f"計算結果: {len(updated_ids)}/{len(bulk_ids)} 件の計算が完了")
    print(f"残り未計算: {len(bulk_ids) - len(updated_ids)} 件")

    # 一時ディレクトリの処理
    if not args.keep_temp:
        try:
            # shutil.rmtree(temp_dir)
            print(f"一時ディレクトリを削除しました: {temp_dir}")
        except Exception as e:
            print(f"警告: 一時ディレクトリの削除に失敗しました: {e}")
    else:
        print(f"一時ディレクトリを保持しました: {temp_dir}")

    print("処理完了")

if __name__ == "__main__":
    main()