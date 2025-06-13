#!/usr/bin/env python3
# filepath: run_120_calculations.py

"""
run_uncalculated.pyを最大120回実行するスクリプト
"""
import subprocess
import time
import sys
import re

SCRIPT_PATH = "/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae/run_uncalculated.py"
MAX_COUNT = 128
WAIT_TIME = 2  # 各実行間の待機時間（秒）

success_count = 0

print(f"run_uncalculated.pyを最大{MAX_COUNT}回実行します")

for i in range(1, MAX_COUNT + 1):
    print(f"-------------------------------------")
    print(f"実行 {i} / {MAX_COUNT} (成功: {success_count})")
    
    # スクリプト実行と出力キャプチャ
    process = subprocess.Popen(
        ["python3", SCRIPT_PATH], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd="/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/ccvae" 
    )
    stdout, stderr = process.communicate()
    
    # 出力を表示
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    
    # 「未計算の構造はありません」というメッセージを検出
    if "未計算の構造はありません" in stdout:
        print("すべての構造が計算済みのため、処理を終了します")
        break
    
    # 成功したかを確認
    if process.returncode == 0:
        if re.search(r"の計算が完了しました", stdout):
            success_count += 1
            print(f"成功回数: {success_count}")
    else:
        print(f"エラーが発生しました (終了コード: {process.returncode})")
    
    # 次の実行まで待機
    if i < MAX_COUNT:
        time.sleep(WAIT_TIME)

print(f"-------------------------------------")
print(f"処理完了: {success_count}/{MAX_COUNT}件の計算が成功しました")