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
from ase.db import connect

# グローバル変数でiter番号を設定
ITER = 1  # ここで現在のiter番号を設定

# 入力ファイル指定（動的に生成）
base_data_dir = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae_2/data"
bulk_file = os.path.join(base_data_dir, f"iter{ITER}_structures.json")
calculation_result_file = os.path.join(base_data_dir, f"iter{ITER}_calculation_result.json")

print(f"=== 未計算構造の過電圧計算 (iter{ITER}) ===")
print(f"構造ファイル: {bulk_file}")
print(f"結果ファイル: {calculation_result_file}")

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
    print(f"Error: bulk DB に接続できませんでした({e})")
    sys.exit(1)

bulk_ids = []
for row in db.select():
    if hasattr(row, 'unique_id') and row.unique_id is not None:
        bulk_ids.append(str(row.unique_id))
    else:
        # フィールド unique_id がなければ id を使用
        bulk_ids.append(str(row.id))

# calculation_result_file の読み込み
with open(calculation_result_file, 'r') as f:
    reaction_data = json.load(f)
reaction_ids = [str(entry.get('unique_id', entry.get('id'))) for entry in reaction_data]

# 最初の未計算 unique_id を取得
uncalculated = [uid for uid in bulk_ids if uid not in reaction_ids]
if not uncalculated:
    print("未計算の構造はありません。")
    sys.exit(0)

uid = uncalculated[0]
print(f"未計算の構造: unique_id={uid}. 計算を開始します...")

# 一時ディレクトリを動的に生成
temp_dir = f"/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae_2/result/test/iter{ITER}_{uid}"
os.makedirs(temp_dir, exist_ok=True)
print(f"一時ディレクトリを作成しました: {temp_dir}")

# ORR_catalyst_generator/code/eta_from_json.py を呼び出して計算
cmd = [
    "python3",
    "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae_2/02_calculate_overpotentials.py",  
    "--bulk_db", bulk_file,
    "--unique_id", uid,
    "--out_json", calculation_result_file,
    "--base_dir", temp_dir,  # 一時ディレクトリを使用
    "--force", "True",  # force オプションに値を追加
    "--log_level", "INFO",  # ログレベルを INFO に設定
    "--calc_type", "mace"  # 計算タイプを指定
]

print(f"Calculating unique_id={uid}...")
result = subprocess.run(cmd)
if result.returncode != 0:
    print(f"Error: unique_id={uid} の計算に失敗しました (exit code {result.returncode})")
    sys.exit(result.returncode)
else:
    print(f"unique_id={uid} の計算が完了しました。")

# 計算後、calculation_result_file を再読み込みして成功数を表示
with open(calculation_result_file, 'r') as f:
    updated = json.load(f)
updated_ids = [str(entry.get('unique_id', entry.get('id'))) for entry in updated]
success = updated_ids.count(uid)

print(f"{len(updated_ids)}/{len(bulk_ids)} 件の計算が成功しました。(計算済みunique_id={uid})")

# 一時ディレクトリを削除
try:
    #shutil.rmtree(temp_dir)
    print(f"一時ディレクトリを削除しました: {temp_dir}")
except Exception as e:
    print(f"警告: 一時ディレクトリの削除に失敗しました: {e}")